import torch
from datasets import load_dataset
from transformers import BertTokenizer

#print("Imporing...")
# Loading a dataset from HuggingFace Datasets library
dataset = load_dataset("rotten_tomatoes")

# Initializing a tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing and preparing the dataset for PyTorch
def tokenize_function(example):
  # Tokenizes the text, applies padding/truncation, and returns tensors
  # including the attention mask.
  return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# The attention mask helps the model distinguish between
# actual data and padding.
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Creating PyTorch DataLoader
train_dataloader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=8, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=8)

from transformers import BertForSequenceClassification
from torch.optim import AdamW

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
# Optimizer and learning rate scheduler setup
optimizer = AdamW(model.parameters(), lr=5e-5)
print("Preparing training...")