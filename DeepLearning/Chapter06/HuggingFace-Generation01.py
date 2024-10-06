import torch
from transformers import pipeline

# Check what version of PyTorch is installed
print(torch.__version__)

# Check the current CUDA version being used
print("CUDA Version: ", torch.version.cuda)

# Check if CUDA is available and if so, print the device name
print("Device name:", torch.cuda.get_device_properties("cuda").name)

# Check if FlashAttention is available
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Before you run this example. Run the following in your terminal.
# "$ pip install packaging"
# "$ pip install ninja"
# "$ pip install flash-attn --no-build-isolation"
# Then run "$ pip install -e ."

# Check device in Pytorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get text generator
generator = pipeline("text-generation", model="gpt2", device=device)
input = "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"

print(generator(input, max_length=256, truncation=True, pad_token_id=generator.tokenizer.eos_token_id))
#print(generator(input, pad_token_id=generator.tokenizer.eos_token_id))
#print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5, pad_token_id=generator.tokenizer.eos_token_id))
#print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5, pad_token_id=generator.tokenizer.eos_token_id))