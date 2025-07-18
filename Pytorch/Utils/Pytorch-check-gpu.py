import torch

# Check GPU availability and PyTorch version
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current GPU device:", torch.cuda.current_device())
