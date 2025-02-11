# test_refined.pyのロード設定（より aggressive な最適化）
load_config = {
    "device_map": "auto",
    "torch_dtype": torch.float16,  # 強制的にfloat16を使用
    "trust_remote_code": True,
    "token": hf_token,
    "low_cpu_mem_usage": True,
    "max_memory": {
        0: "12GB",  # 固定的なメモリ割り当て
        1: "12GB",
        "cpu": "24GB"
    }
} 