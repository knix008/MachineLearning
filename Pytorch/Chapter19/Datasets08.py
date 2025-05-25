import torch
from datasets import load_dataset
from transformers import BertTokenizer

# Example: Using GPU for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

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
model.to(device)
# Optimizer and learning rate scheduler setup
optimizer = AdamW(model.parameters(), lr=5e-5)

from tqdm import tqdm

# Training loop using PyTorch
for epoch in range(3):  # Train for 3 epochs as an example
    model.train()
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        input_ids = input_ids.to(device)
        attention_mask = batch['attention_mask']
        attention_mask = attention_mask.to(device)
        labels = batch['label']
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluation loop
    model.eval()
    total_correct = 0
    total_samples = 0
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            input_ids = batch['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.to(device)
            labels = batch['label']
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

    accuracy = total_correct / total_samples
    print(f"Epoch {epoch + 1} - Evaluation Accuracy: {accuracy}")