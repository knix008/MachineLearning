import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset

import warnings 
# Suppress warnings
warnings.filterwarnings(action='ignore') 

# 1. Penn Treebank Dataset Loader (word-level)
class PTBDataset(Dataset):
    def __init__(self, texts, seq_len, word2idx, idx2word):
        self.seq_len = seq_len
        self.word2idx = word2idx
        self.idx2word = idx2word
        # Flatten sentences to a single sequence of word indices
        words = [w for sentence in texts for w in sentence.split()]
        self.data = [word2idx[w] for w in words if w in word2idx]
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# 2. Build vocabulary from PTB train split
def build_vocab_ptb(texts, vocab_size=10000):
    from collections import Counter
    words = [w for sentence in texts for w in sentence.split()]
    most_common = Counter(words).most_common(vocab_size-2)
    idx2word = ["<PAD>", "<UNK>"] + [w for w, _ in most_common]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word

# 3. Basic Transformer Model (unchanged)
class BasicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, max_seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x: (batch, seq_len)
        b, seq_len = x.size()
        emb = self.embed(x) + self.pos_embed[:, :seq_len, :]
        emb = emb.transpose(0, 1)  # Transformer expects (seq_len, batch, d_model)
        output = self.transformer(emb)
        output = output.transpose(0, 1)  # (batch, seq_len, d_model)
        logits = self.fc(output)  # (batch, seq_len, vocab)
        return logits

# 4. Training loop
def train(model, dataloader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        losses = []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}")

# 5. Text generation (word-level)
def generate_text(model, start_seq, word2idx, idx2word, length=50, device="cpu"):
    model.eval()
    seq = [word2idx.get(w, word2idx["<UNK>"]) for w in start_seq.split()]
    for _ in range(length):
        inp = torch.tensor([seq[-model.max_seq_len:]], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(inp)
        next_id = torch.argmax(logits[0, -1]).item()
        seq.append(next_id)
    return ' '.join([idx2word[i] for i in seq])

# 6. Main
if __name__ == "__main__":
    # Load PTB dataset (train split)
    ptb = load_dataset("ptb_text_only", split="train")
    texts = ptb["sentence"]  # list of sentences

    seq_len = 32
    vocab_size = 5000
    word2idx, idx2word = build_vocab_ptb(texts, vocab_size=vocab_size)
    dataset = PTBDataset(texts, seq_len, word2idx, idx2word)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicTransformer(vocab_size=len(word2idx), max_seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train(model, dataloader, optimizer, criterion, device, epochs=5)
    
    # Generate text
    prompt = "the company said"
    generated = generate_text(model, prompt, word2idx, idx2word, length=50, device=device)
    print("Generated Text:")
    print(generated)