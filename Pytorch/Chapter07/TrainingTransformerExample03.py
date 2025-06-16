import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nltk
import os

nltk.download("gutenberg")
from nltk.corpus import gutenberg
from collections import Counter


class Vocab:
    def __init__(self, tokens, min_freq=1, reserved_tokens=["<pad>", "<unk>"]):
        counter = Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = reserved_tokens.copy()
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                continue
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def to_indices(self, tokens):
        return [
            self.token_to_idx.get(token, self.token_to_idx["<unk>"]) for token in tokens
        ]

    def to_tokens(self, indices):
        return [self.idx_to_token[idx] for idx in indices]


def get_tokens_and_vocab():
    text = gutenberg.raw("austen-emma.txt")
    tokens = nltk.word_tokenize(text)
    vocab = Vocab(tokens, min_freq=2)
    data = torch.tensor(vocab.to_indices(tokens), dtype=torch.long)
    return data, vocab


class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            self._generate_positional_encoding(max_seq_len, embed_dim),
            requires_grad=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def _generate_positional_encoding(self, max_seq_len, embed_dim):
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_seq_len, embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        out = self.transformer_encoder(x)
        out = self.fc_out(out)
        return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def save_model(model, vocab, path="transformer_model.pth"):
    """Save model state_dict and vocab."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_idx_to_token": (
            model.vocab.idx_to_token if hasattr(model, "vocab") else vocab.idx_to_token
        ),
        "vocab_token_to_idx": (
            model.vocab.token_to_idx if hasattr(model, "vocab") else vocab.token_to_idx
        ),
        "hyperparams": {
            "vocab_size": model.embedding.num_embeddings,
            "embed_dim": model.embedding.embedding_dim,
            "num_heads": model.transformer_encoder.layers[0].self_attn.num_heads,
            "num_layers": len(model.transformer_encoder.layers),
        },
    }
    torch.save(checkpoint, path)
    print(f"Model and vocab saved to {path}.")


def load_model(path="transformer_model.pth", device="cpu"):
    """Load model state_dict and vocab."""
    checkpoint = torch.load(path, map_location=device)
    vocab_size = checkpoint["hyperparams"]["vocab_size"]
    embed_dim = checkpoint["hyperparams"]["embed_dim"]
    num_heads = checkpoint["hyperparams"]["num_heads"]
    num_layers = checkpoint["hyperparams"]["num_layers"]
    model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    # Reconstruct vocab
    vocab = Vocab([])
    vocab.idx_to_token = checkpoint["vocab_idx_to_token"]
    vocab.token_to_idx = checkpoint["vocab_token_to_idx"]
    print(f"Model and vocab loaded from {path}.")
    return model, vocab


def generate_text(
    model, vocab, seed_text, generate_len=50, device="cpu", temperature=1.0, seq_len=32
):
    model.eval()
    tokens = nltk.word_tokenize(seed_text)
    indices = vocab.to_indices(tokens)
    generated = indices.copy()
    for _ in range(generate_len):
        input_seq = torch.tensor(
            [generated[-seq_len:]], dtype=torch.long, device=device
        )
        with torch.no_grad():
            logits = model(input_seq)
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
        generated.append(next_idx)
    gen_tokens = vocab.to_tokens(generated)
    return " ".join(gen_tokens)


def main(
    seq_len=32,
    batch_size=64,
    epochs=15,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    save_path="transformer_model.pth",
    do_train=True,
    do_generate=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if do_train or not os.path.exists(save_path):
        data, vocab = get_tokens_and_vocab()
        n = len(data) - seq_len
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        train_data = data[: train_end + seq_len]
        val_data = data[train_end : val_end + seq_len]
        test_data = data[val_end:]

        train_set = SequenceDataset(train_data, seq_len)
        val_set = SequenceDataset(val_data, seq_len)
        test_set = SequenceDataset(test_data, seq_len)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        model = TransformerModel(
            len(vocab), embed_dim, num_heads, num_layers, max_seq_len=seq_len
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print(
            f"Vocab size: {len(vocab)} | Train batches: {len(train_loader)} | Device: {device}"
        )
        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = evaluate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        # Attach vocab to model for saving convenience
        model.vocab = vocab
        save_model(model, vocab, save_path)
    else:
        model, vocab = load_model(save_path, device=device)

    if do_generate:
        seed_text = "Emma Woodhouse, handsome, clever, and rich,"
        print("\nGenerating text:")
        generated = generate_text(
            model, vocab, seed_text, generate_len=5, device=device, seq_len=seq_len
        )
        print(generated)


if __name__ == "__main__":
    main()
