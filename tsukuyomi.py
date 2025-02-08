import os
import pandas as pd
import torch
from datasets import Dataset
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# huggingfaceトークンの設定（gemma2を使用するのに必要なため）
os.environ["HF_TOKEN"] = ""

# モデルのリポジトリIDを設定
repo_id = "google/gemma-2-2b-it"

# データセットのパス
dataset_path = "./つくよみちゃん会話AI育成計画.csv"

# csvファイルを読み込む
csv_data = pd.read_csv(dataset_path, skiprows=1)

# Datasetオブジェクトにcsvデータを変換
dataset = Dataset.from_pandas(csv_data)

# プロンプトフォーマット
PROMPT_FORMAT = """<start_of_turn>user
あなたは"つくよみちゃん"です。礼儀正しく、健気で優しい女の子です。

{instruction_1}
<end_of_turn>
<start_of_turn>model
{output_1}
<end_of_turn>
"""

# プロンプトフォーマット（追加）
ADD_PROMPT_FORMAT = """<start_of_turn>user
{instruction_2}
<end_of_turn>
<start_of_turn>model
{output_2}
<end_of_turn>
"""

# データセットの内容をプロンプトにセット → textフィールドとして作成する関数
def generate_text_field(examples):
    instruction_1 = examples["【A】話しかけ"]
    output_1 = examples["【B】お返事"]
    instruction_2 = examples.get("【C】Bに対するA話者の返事（ある場合のみ）", "")
    output_2 = examples.get("【D】Cに対するつくよみちゃんのお返事（ある場合のみ）", "")
    full_prompt = PROMPT_FORMAT.format(
        instruction_1=instruction_1,
        output_1=output_1
    )
    if instruction_2 and output_2:
        full_prompt += ADD_PROMPT_FORMAT.format(
            instruction_2=instruction_2,
            output_2=output_2
        )
    return {"text": full_prompt}

# データセットに（generate_text_fieldの処理を用いて）textフィールドを追加
train_dataset = dataset.map(generate_text_field)

# 量子化のConfigを設定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # 4ビット量子化を使用
    bnb_4bit_quant_type="nf4", # 4ビット量子化の種類にnf4（NormalFloat4）を使用
    bnb_4bit_use_double_quant=True, # 二重量子化を使用
    bnb_4bit_compute_dtype=torch.float16 # 量子化のデータ型をfloat16に設定
)

# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=repo_id, # モデルのリポジトリIDをセット
    device_map={"": "cuda"}, # 使用デバイスを設定
    quantization_config=quantization_config, # 量子化のConfigをセット
    attn_implementation="eager", # 注意機構に"eager"を設定（Gemma2モデルの学習で推奨されているため）
)

# キャッシュを無効化（メモリ使用量を削減）
model.config.use_cache = False 

# テンソル並列ランクを１に設定（テンソル並列化を使用しない）
model.config.pretraining_tp = 1 

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=repo_id, # モデルのリポジトリIDをセット
    attn_implementation="eager", # 注意機構に"eager"を設定（Gemma2モデルの学習で推奨されているため）
    add_eos_token=True, # EOSトークンの追加を設定
)

# パディングトークンが設定されていない場合、EOSトークンを設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# パディングを右側に設定(fp16を使う際のオーバーフロー対策)
tokenizer.padding_side = "right"

# モデルから（4ビット量子化された）線形層の名前を取得する関数
def find_all_linear_names(model):
    target_class = bnb.nn.Linear4bit
    linear_layer_names = set()
    for name_list, module in model.named_modules():
        if isinstance(module, target_class):
            names = name_list.split('.')
            layer_name = names[-1] if len(names) > 1 else names[0]
            linear_layer_names.add(layer_name)
    if 'lm_head' in linear_layer_names:
        linear_layer_names.remove('lm_head')
    return list(linear_layer_names)

# 線形層の名前を取得
target_modules = find_all_linear_names(model)

# LoRAのConfigを設定
Lora_config = LoraConfig(
    lora_alpha=8, # LoRAによる学習の影響力を調整（スケーリング)
    lora_dropout=0.1, # ドロップアウト率
    r=4, # 低ランク行列の次元数
    bias="none", # バイアスのパラメータ更新
    task_type="CAUSAL_LM", # タスクの種別
    target_modules=target_modules # LoRAを適用するモジュール
)

# 学習パラメータを設定
training_arguments = TrainingArguments(
    output_dir="./train_logs", # ログの出力ディレクトリ
    fp16=True, # fp16を使用
    logging_strategy='epoch', # 各エポックごとにログを保存（デフォルトは"steps"）
    save_strategy='epoch', # 各エポックごとにチェックポイントを保存（デフォルトは"steps"）
    num_train_epochs=3, # 学習するエポック数
    per_device_train_batch_size=1, # （GPUごと）一度に処理するバッチサイズ
    gradient_accumulation_steps=4, # 勾配を蓄積するステップ数
    optim="paged_adamw_32bit", # 最適化アルゴリズム
    learning_rate=1e-4, # 初期学習率
    lr_scheduler_type="cosine", # 学習率スケジューラの種別
    max_grad_norm=0.3, # 勾配の最大ノルムを制限（クリッピング）
    warmup_ratio=0.03, # 学習を増加させるウォームアップ期間の比率
    weight_decay=0.001, # 重み減衰率
    group_by_length=True,# シーケンスの長さが近いものをまとめてバッチ化
    report_to="tensorboard" # TensorBoard使用してログを生成（"./train_logs"に保存）
)

# SFTパラメータの設定
trainer = SFTTrainer(
    model=model, # モデルをセット
    tokenizer=tokenizer, # トークナイザーをセット
    train_dataset=train_dataset, # データセットをセット
    dataset_text_field="text", # 学習に使用するデータセットのフィールド
    peft_config=Lora_config, # LoRAのConfigをセット
    args=training_arguments, # 学習パラメータをセット
    max_seq_length=512, # 入力シーケンスの最大長を設定
)

# 正規化層をfloat32に変換(学習を安定させるため)
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# モデルの学習
trainer.train()

# 学習したアダプターを保存
trainer.model.save_pretrained("./つくよみちゃん_Adapter")