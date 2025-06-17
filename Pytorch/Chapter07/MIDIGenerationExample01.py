import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from mido import MidiFile, MidiTrack, Message
import glob
import os


def midi_to_note_sequence(midi_file):
    """Extract note numbers from midi file, ignoring velocity and time for simplicity."""
    midi = MidiFile(midi_file)
    notes = []
    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta and msg.type == "note_on" and msg.velocity > 0:
                notes.append(msg.note)
    return notes


def note_sequence_to_midi(note_sequence, output_file, tempo=500000):
    """Convert note number sequence to a simple MIDI file."""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message("program_change", program=0, time=0))
    for note in note_sequence:
        track.append(Message("note_on", note=int(note), velocity=64, time=0))
        track.append(Message("note_off", note=int(note), velocity=64, time=120))
    mid.save(output_file)


class MidiNoteDataset(Dataset):
    def __init__(self, midi_folder, seq_len=50):
        self.seq_len = seq_len
        self.notes = []
        midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + glob.glob(
            os.path.join(midi_folder, "*.midi")
        )
        for mf in midi_files:
            ns = midi_to_note_sequence(mf)
            self.notes.extend(ns)
        self.notes = np.array(self.notes)
        self.unique_notes = sorted(list(set(self.notes)))
        self.note2idx = {n: i for i, n in enumerate(self.unique_notes)}
        self.idx2note = {i: n for i, n in enumerate(self.unique_notes)}
        self.data = [self.note2idx[n] for n in self.notes]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_len]
        target = self.data[idx + self.seq_len]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def generate_notes(model, start_seq, length, idx2note, device, temperature=1.0):
    model.eval()
    generated = list(start_seq)
    input_seq = torch.tensor(start_seq, dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    for _ in range(length):
        out, hidden = model(input_seq, hidden)
        out = out / temperature
        prob = torch.softmax(out, dim=-1)
        next_idx = torch.multinomial(prob, 1).item()
        generated.append(next_idx)
        input_seq = torch.cat(
            [input_seq[:, 1:], torch.tensor([[next_idx]], device=device)], dim=1
        )
    return [idx2note[idx] for idx in generated]


def main(
    midi_folder="midi_dataset",
    output_midi="generated.mid",
    seq_len=50,
    batch_size=64,
    epochs=20,
    embed_dim=128,
    hidden_dim=256,
    num_layers=2,
    lr=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Prepare Dataset
    dataset = MidiNoteDataset(midi_folder, seq_len=seq_len)
    vocab_size = len(dataset.unique_notes)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model
    model = MusicLSTM(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
    print(f"> Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Test
    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"> Test Loss: {test_loss:.4f}")

    # Generate new music
    start_seq = train_set.dataset.data[:seq_len]
    generated_notes = generate_notes(model, start_seq, 200, dataset.idx2note, device)
    note_sequence_to_midi(generated_notes, output_midi)
    print(f"> Generated MIDI saved to {output_midi}")


if __name__ == "__main__":
    main(
        midi_folder="./data/Nottingham/train",
        output_midi="generated.mid",
    )
