import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import movie_reviews
from torch.utils.data import Dataset, DataLoader
import random

import warnings

warnings.filterwarnings("ignore")  # To ignore Userwarning messages.

# Download NLTK data
nltk.download("movie_reviews")


# 1. Prepare dataset
def get_data():
    docs = []
    labels = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            words = list(movie_reviews.words(fileid))
            docs.append(words)
            labels.append(1 if category == "pos" else 0)
    return docs, labels


docs, labels = get_data()

# Shuffle and split
combined = list(zip(docs, labels))
random.shuffle(combined)
split = int(0.8 * len(combined))
train_data = combined[:split]
test_data = combined[split:]

# 2. Build vocabulary
from collections import Counter

MAX_VOCAB_SIZE = 10000
UNK_IDX = 0
PAD_IDX = 1
EOS_IDX = 2

vocab = ["<unk>", "<pad>", "<eos>"]
counter = Counter(word.lower() for doc, _ in train_data for word in doc)
vocab += [word for word, _ in counter.most_common(MAX_VOCAB_SIZE - len(vocab))]
word2idx = {word: idx for idx, word in enumerate(vocab)}


def encode(doc):
    ids = [word2idx.get(word.lower(), UNK_IDX) for word in doc]
    ids.append(EOS_IDX)
    return ids


# 3. Dataset class with EOS and correct padding
MAX_LEN = 200


class MovieReviewDataset(Dataset):
    def __init__(self, data):
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        doc, label = self.samples[idx]
        ids = encode(doc)
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        else:
            ids += [PAD_IDX] * (MAX_LEN - len(ids))
        # Padding mask: 1 for pad, 0 for tokens
        pad_mask = [1 if token == PAD_IDX else 0 for token in ids]
        return torch.tensor(ids), torch.tensor(pad_mask), torch.tensor(label)


def collate_fn(batch):
    inputs, masks, labels = zip(*batch)
    return torch.stack(inputs), torch.stack(masks), torch.tensor(labels)


train_dataset = MovieReviewDataset(train_data)
test_dataset = MovieReviewDataset(test_data)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


# 4. Transformer model with padding mask
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask):
        # x: [batch, seq_len], pad_mask: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, d_model]
        # src_key_padding_mask expects [batch, seq_len] with True for PAD
        out = self.transformer(emb, src_key_padding_mask=pad_mask.bool())
        out = out.mean(dim=1)  # mean pooling over seq_len
        return self.fc(out)


# 5. Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 15 # Default is 5

def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, pad_mask, targets in train_loader:
            inputs, pad_mask, targets = (
                inputs.to(device),
                pad_mask.to(device),
                targets.to(device),
            )
            outputs = model(inputs, pad_mask)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")


def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, pad_mask, targets in test_loader:
            inputs, pad_mask, targets = (
                inputs.to(device),
                pad_mask.to(device),
                targets.to(device),
            )
            outputs = model(inputs, pad_mask)
            preds = outputs.argmax(1)
            preds = preds.to(device)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    print(f"Test accuracy: {correct/total:.4f}")


# 6. Prediction on example input
def preprocess_text(text):
    return [w.lower() for w in text.split()]


def predict_sentiment(text):
    model.eval()
    tokens = preprocess_text(text)
    ids = encode(tokens)
    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
    else:
        ids += [PAD_IDX] * (MAX_LEN - len(ids))
    pad_mask = [1 if token == PAD_IDX else 0 for token in ids]
    input_tensor = torch.tensor([ids]).to(device)
    pad_mask_tensor = torch.tensor([pad_mask]).to(device)
    with torch.no_grad():
        output = model(input_tensor, pad_mask_tensor)
        pred = output.argmax(1).item()
        sentiment = "positive" if pred == 1 else "negative"
    print(f"Input: {text}\nPredicted sentiment: {sentiment}\n")


if __name__ == "__main__":
    train()
    evaluate()

    # Example inputs
    print("> Sentiment Analysis Example Predictions :")
    predict_sentiment("This is the best movie I have ever seen. Absolutely wonderful and touching.")
    predict_sentiment("What a waste of time. The plot was terrible and the acting was worse.")
    predict_sentiment("> It was okay, not bad but nothing special.")
