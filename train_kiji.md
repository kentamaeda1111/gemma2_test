

### 高優先度（面接官が非常に興味を持ちそうな点）

1. **カスタムアテンションマスク処理**
```python
def preprocess_function(examples):
    # Pattern definitions
    end_patterns = ["だろうか", "ではないか", "のではないか"...]
    conjunctions = ["しかし", "だから", "それでは"...]
    # ...
```
- なぜ特定のパターンに異なる重みを付けたのか
- この重み付けが学習結果にどう影響するのか

2. **日本語特有の前処理**

```python
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '。', '、', '！', '？',  # Punctuation marks
    ]
})
```
- 日本語特有のトークン化の課題への対応
- 句読点の取り扱いについての工夫

3. **評価データセットの設計**

```python
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+100])
```

- なぜ評価用データセットを100サンプルに制限したのか
- どのように評価の質を担保しているのか

### 中優先度（質問される可能性がある点）
1. **トレーニングモニタリングシステム**
```python
class TrainingMonitorCallback(TrainerCallback):
    # ...
    def _plot_metrics(self):
        # ...
```
- どのようにトレーニングの進捗を監視したか
- 異常検知の方法

2. **LoRA設定**
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # ...
)
```
- パラメータ選択の根拠
- ファインチューニング戦略

3. **4ビット量子化の実装**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # ...
)
```
- 量子化による精度への影響
- メモリ使用量の最適化方法

1. **ハイパーパラメータ設定**
```python
training_args = TrainingArguments(
    num_train_epochs=30,
    learning_rate=8e-5,
    weight_decay=0.06,
    # ...
)
```
- パラメータチューニングの方法
- 最適化の過程
