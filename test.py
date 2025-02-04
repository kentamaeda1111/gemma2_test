import torch
import bitsandbytes as bnb

if torch.cuda.is_available():
    print("CUDA is available")
    major, minor = torch.cuda.get_device_capability()
    print(f"CUDA Device Capability: {major}.{minor}")

    try:
        # Force initialization
        bnb.cuda_setup.initialize()
        print("Bitsandbytes has been initialized")

        if bnb.cuda_setup.initialized:
            print("Bitsandbytes is initialized")
            print(f"Bitsandbytes CUDA version: {bnb.cuda_setup.version}")
            print(f"Bitsandbytes CUDA library path: {bnb.cuda_setup.cuda_dll_path}")
        else:
            print("Bitsandbytes is not initialized")

    except Exception as e:
        print(f"Error during bitsandbytes check: {e}")

else:
    print("CUDA is not available")