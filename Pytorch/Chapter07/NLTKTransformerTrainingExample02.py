import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import numpy as np

# Download required NLTK resources
nltk.download("punkt_tab")

# 1. Data Preparation
text = "This is an example sentence. Another sentence here."
tokens = word_tokenize(text.lower())
vocab = FreqDist(tokens)
vocab_size = len(vocab)
token_to_index = {token: i for i, token in enumerate(vocab.keys())}
index_to_token = {i: token for token, i in token_to_index.items()}

# Convert tokens to indices
indexed_tokens = [token_to_index[token] for token in tokens]

# Create input and target sequences (example: shift by one)
input_seqs = indexed_tokens[:-1]
target_seqs = indexed_tokens[1:]

# Convert to tensors
input_tensor = torch.tensor(input_seqs).unsqueeze(0)
target_tensor = torch.tensor(target_seqs).unsqueeze(0)


# 2. Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_mask=None):
        src = self.embedding(src)
        output = self.transformer(src, src, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output


# 3. Training
model = TransformerModel(vocab_size)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop (simplified)
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output.squeeze(0), target_tensor.squeeze(0))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# Inference
model.eval()
with torch.no_grad():
    input_seq = torch.tensor([token_to_index["example"]]).unsqueeze(0)
    output = model(input_seq)
    predicted_index = output.argmax(dim=-1).item()
    predicted_token = index_to_token[predicted_index]
    print(f"> Input: example, Prediction: {predicted_token}")
