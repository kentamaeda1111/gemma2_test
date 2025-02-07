
# Optimize dataset processing
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=16,
    num_proc=2,
    load_from_cache_file=True,
    desc="Tokenizing datasets",
    remove_columns=dataset.column_names,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="balanced",
    torch_dtype=torch.float16,
    attn_implementation='sdpa',
    token=os.environ["HUGGINGFACE_TOKEN"],  
    max_memory={0: "4GiB", 1: "4GiB", "cpu": "24GB"}
)

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,  
    num_train_epochs=30,
    learning_rate=8e-5,
    weight_decay=0.06,
    warmup_ratio=0.25,
    lr_scheduler_type="cosine_with_restarts",
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    gradient_accumulation_steps=8,
    max_steps=-1,
    disable_tqdm=False,
    logging_dir=LOG_OUTPUT_DIR,   
    logging_strategy="steps",
    logging_steps=50,
    no_cuda=False,
    dataloader_num_workers=1,
    report_to=[],
    run_name=None,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    dataloader_pin_memory=True,
    save_total_limit=2,
    fp16=True,
    optim="adamw_torch_fused",
    eval_accumulation_steps=4,
    load_best_model_at_end=True,
    metric_for_best_model="combined_score",
)

