import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import transformers as transformers
from transformers import pipeline
print("Transformer version : ", transformers.__version__)

from transformers import logging
logging.set_verbosity_error()

import torch 
print("Pytorch version :", torch.__version__)
# Check device in Pytorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ner_pipe = pipeline("ner", model='bert-large-cased', device=device)
sequence = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much."""

for entity in ner_pipe(sequence):
    print(entity) 
