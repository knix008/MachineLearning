import os
from transformers import AutoTokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sequence = "The original theory of relativity is based upon the premise that all coordinate systems"
print(tokenizer(sequence))