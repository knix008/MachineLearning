import torch
from datasets import load_dataset
from transformers import BertTokenizer
from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator(cpu=False, mixed_precision="fp16")

# Loading a dataset from HuggingFace Datasets library
dataset = load_dataset("rotten_tomatoes")

# Initializing a tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing and preparing the dataset for PyTorch
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Creating PyTorch DataLoader
train_dataloader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=8, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=8)

from tqdm import tqdm
from transformers import BertForSequenceClassification
from torch.optim import AdamW

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Move model to the device managed by Accelerator
model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)
print("> Instantiating model and dataloader...")