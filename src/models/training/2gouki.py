# トークナイザーの設定を追加
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '。', '、', '！', '？',  # 句読点
    ]
})

# 重複したtokenize_dataset.map()を削除し、1回だけにする
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,
    num_proc=4,
    load_from_cache_file=True,
    desc="Tokenizing datasets",
    remove_columns=dataset.column_names,
)

# データセットの検証
tokenized_dataset = validate_dataset(tokenized_dataset) 