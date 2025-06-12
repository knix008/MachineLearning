import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize

# Sample data
text = "This is an example sentence. Another sentence here."
nltk.download("punkt_tab")
tokens = word_tokenize(text.lower())

# Vocabulary
vocab = set(tokens)
vocab_size = len(vocab)
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

# Numericalize the tokens
input_ids = [word_to_id[token] for token in tokens]
input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension


# Transformer Model (simplified example)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, batch_first=True)
        self.fc = nn.Linear(
            d_model, vocab_size
        )  # output layer for next word prediction

    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer(
            embedded, embedded
        )  # Simplified, no separate encoder/decoder
        output = self.fc(output)
        return output


# Model parameters
d_model = 64
nhead = 2
num_layers = 2
model = TransformerModel(vocab_size, d_model, nhead, num_layers)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(input_tensor)
    target = input_tensor.roll(-1, dims=1)  # Shifted input to predict next word
    loss = criterion(output.squeeze(0), target.squeeze(0))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
