import torch
from transformers import pipeline
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Example: Using GPU for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

pipeline = pipeline(
    task="fill-mask",
    model="google-bert/bert-base-uncased",
    torch_dtype=torch.float16,
    device=device
)

result = pipeline("Plants create [MASK] through a process known as photosynthesis.")
print(result[0])