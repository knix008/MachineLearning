import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_mask=None, memory_mask=None):
        src_embedded = self.embedding(src) * torch.sqrt(
            torch.tensor(self.embedding.embedding_dim)
        )
        tgt_embedded = self.embedding(tgt) * torch.sqrt(
            torch.tensor(self.embedding.embedding_dim)
        )

        src_embedded = self.positional_encoding(src_embedded)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        encoder_output = self.encoder(src_embedded, mask=src_mask)
        decoder_output = self.decoder(
            tgt_embedded, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

        output = self.linear(decoder_output)
        return output


def main():
    # Example usage:
    vocab_size = 1000
    d_model = 512
    nhead = 8
    num_layers = 6

    # Define a simple transformer model
    model = Transformer(vocab_size, d_model, nhead, num_layers)

    # Sample input sequences (source and target)
    src_sequence = torch.randint(0, vocab_size, (10, 10))  # (batch_size, src_len)
    tgt_sequence = torch.randint(0, vocab_size, (10, 10))  # (batch_size, tgt_len)

    # Create dummy masks (optional, but often needed)
    src_mask = torch.zeros(src_sequence.shape[0], src_sequence.shape[1]).bool()
    tgt_mask = torch.zeros(tgt_sequence.shape[0], tgt_sequence.shape[1]).bool()
    memory_mask = None

    # Pass the data through the model
    output = model(
        src_sequence,
        tgt_sequence,
        tgt_mask=tgt_mask,
        src_mask=src_mask,
        memory_mask=memory_mask,
    )

    # Output: (batch_size, tgt_len, vocab_size) -  logits for each token in the target sequence
    print(output.shape)  # Expected output: (10, 6, 1000)


if __name__ == "__main__":
    main()
