import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from music21 import converter, instrument, note, chord


def extract_notes(midi_folder):
    """Extract notes and chords from all MIDI files in a folder."""
    notes = []
    for file in os.listdir(midi_folder):
        if file.endswith(".mid") or file.endswith(".midi"):
            midi_path = os.path.join(midi_folder, file)
            midi = converter.parse(midi_path)
            parts = instrument.partitionByInstrument(midi)
            if parts:  # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else:  # file has notes in flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
    return notes


def extract_notes_from_midi_files():
    # Directories for datasets
    train_dir = "./data/Nottingham/train"
    val_dir = "./data/Nottingham/valid"
    test_dir = "./data/Nottingham/test"

    # Extract notes from each split
    notes_train = extract_notes(train_dir)
    notes_val = extract_notes(val_dir)
    notes_test = extract_notes(test_dir)
    return notes_train, notes_val, notes_test


def build_vocabulary(notes_train, notes_val, notes_test):
    """Build vocabulary from all notes."""
    # Build vocabulary from all notes
    all_notes = notes_train + notes_val + notes_test
    pitches = sorted(set(all_notes))
    note_to_int = {note: i for i, note in enumerate(pitches)}
    int_to_note = {i: note for note, i in note_to_int.items()}
    n_vocab = len(note_to_int)
    return note_to_int, int_to_note, n_vocab


def prepare_sequences(notes, note_to_int, sequence_length):
    input_seqs = []
    output_seqs = []
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i : i + sequence_length]
        seq_out = notes[i + sequence_length]
        input_seqs.append([note_to_int[n] for n in seq_in])
        output_seqs.append(note_to_int[seq_out])
    return np.array(input_seqs), np.array(output_seqs)


def load_data(notes_train, notes_val, notes_test, note_to_int, sequence_length):
    """Load and prepare data for training, validation, and testing."""
    X_train, y_train = prepare_sequences(notes_train, note_to_int, sequence_length)
    X_val, y_val = prepare_sequences(notes_val, note_to_int, sequence_length)
    X_test, y_test = prepare_sequences(notes_test, note_to_int, sequence_length)
    return X_train, y_train, X_val, y_val, X_test, y_test


class MIDIDataset(Dataset):
    def __init__(self, X, y, n_vocab):
        self.X = torch.tensor(X / float(n_vocab), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(-1), self.y[idx]


def load_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, n_vocab):
    """Create DataLoader for training, validation, and testing."""
    batch_size = 64
    train_dataset = MIDIDataset(X_train, y_train, n_vocab)
    val_dataset = MIDIDataset(X_val, y_val, n_vocab)
    test_dataset = MIDIDataset(X_test, y_test, n_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out


def train_model(model, loss_fn, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5  # Originally 50, reduced for quicker testing

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                val_loss += loss_fn(val_outputs, val_y).item()

        print(
            f"> Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        )


def evaluate_model(model, loss_fn, test_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            test_outputs = model(test_x)
            test_loss += loss_fn(test_outputs, test_y).item()
    print(f"> Test Loss: {test_loss/len(test_loader):.4f}")


def generate_notes_lstm(
    model, seed_seq, int_to_note, n_vocab, length=100, sequence_length=32, device=None
):
    model.eval()
    generated = []
    pattern = seed_seq.copy()
    for _ in range(length):
        input_seq = np.array(pattern[-sequence_length:]) / float(n_vocab)
        input_seq = (
            torch.tensor(input_seq, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device)
        )
        with torch.no_grad():
            pred = model(input_seq)
            top_i = torch.argmax(pred).item()
        generated.append(int_to_note[top_i])
        pattern.append(top_i)
    return generated


def main():
    print("> Music Generation with LSTM")
    # Extract notes from MIDI files
    notes_train, notes_val, notes_test = extract_notes_from_midi_files()
    note_to_int, int_to_note, n_vocab = build_vocabulary(
        notes_train, notes_val, notes_test
    )
    sequence_length = 32  # Fixed sequence length for LSTM
    print(f"> Vocabulary Size: {n_vocab}, Sequence Length: {sequence_length}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        notes_train, notes_val, notes_test, note_to_int, sequence_length
    )
    train_loader, val_loader, test_loader = load_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, n_vocab
    )
    # Initialize the model
    model = MusicLSTM(input_size=1, hidden_size=256, output_size=n_vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    train_model(model, loss_fn, train_loader, val_loader, device)
    evaluate_model(model, loss_fn, test_loader, device)

    # Example: Generate 100 notes from a random seed in test dataset
    if X_test.shape[0] > 0:
        start_idx = np.random.randint(0, X_test.shape[0] - 1)
        seed = list(X_test[start_idx])
        generated_notes = generate_notes_lstm(
            model,
            seed,
            int_to_note,
            n_vocab,
            length=100,
            sequence_length=sequence_length,
            device=device,
        )
        print("> Generated Notes:", generated_notes)
    else:
        print("> Not enough test data to generate notes.")


if __name__ == "__main__":
    main()
