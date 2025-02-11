# test_nsystemprompt.pyのロード設定
load_config = {
    "trust_remote_code": True,
    "token": hf_token,
    "low_cpu_mem_usage": True
}

# デバイスに応じて設定を追加
if device == "cuda":
    load_config["device_map"] = "auto"
    load_config["torch_dtype"] = torch.bfloat16
else:
    load_config["device_map"] = "auto"
    load_config["torch_dtype"] = torch.float32
    load_config["offload_folder"] = "offload_folder" 