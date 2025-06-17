import numpy as np
import glob
import pickle
from music21 import converter, instrument, note, chord, stream
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 1. MIDI 파일에서 note와 chord 추출
def get_notes(midi_folder="./data/Nottingham/train/"):
    notes = []
    print(f"> Loading MIDI files from {midi_folder}...")
    for file in glob.glob(f"{midi_folder}/*.mid"):
        midi = converter.parse(file)
        # print(f"Parsing {file}")
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))
    return notes


# 2. 데이터 전처리 (시퀀스 생성)
def prepare_sequences(notes, n_vocab, sequence_length=100):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i : i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])
    return network_input, network_output, pitchnames, note_to_int


class MidiDataset(Dataset):
    def __init__(self, network_input, network_output):
        self.X = torch.tensor(network_input, dtype=torch.long)
        self.y = torch.tensor(network_output, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 3. LSTM 모델 생성
class LSTMMusicModel(nn.Module):
    def __init__(
        self, n_vocab, seq_length, embedding_dim=100, hidden_size=256, num_layers=2
    ):
        super(LSTMMusicModel, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, n_vocab)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        if hidden:
            out, hidden = self.lstm(x, hidden)
        else:
            out, hidden = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out, hidden


# 4. 학습 함수
def train(model, loader, epochs, lr, device):
    print("> Starting training...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(loader):.4f}")


# 5. MIDI 생성
def generate_notes(model, network_input, pitchnames, n_vocab, device, num_generate=500):
    model.eval()
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    pattern = torch.tensor(pattern, dtype=torch.long).unsqueeze(0).to(device)
    prediction_output = []
    hidden = None
    for note_index in range(num_generate):
        with torch.no_grad():
            output, hidden = model(pattern, hidden)
        index = torch.argmax(output, dim=1).item()
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = torch.cat(
            [pattern[:, 1:], torch.tensor([[index]], dtype=torch.long).to(device)],
            dim=1,
        )
    return prediction_output


# 6. MIDI 파일로 변환
def create_midi(prediction_output, output_file="nottingham_output_pytorch.mid"):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_file)


# 전체 파이프라인 실행 예시
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    notes = get_notes("./data/Nottingham/train")
    with open("data/notes", "wb") as filepath:
        pickle.dump(notes, filepath)
    n_vocab = len(set(notes))
    sequence_length = 100
    network_input, network_output, pitchnames, note_to_int = prepare_sequences(
        notes, n_vocab, sequence_length
    )
    dataset = MidiDataset(network_input, network_output)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = LSTMMusicModel(n_vocab, sequence_length).to(device)
    train(model, loader, epochs=1, lr=0.001, device=device)  # Adjust epochs as needed
    prediction_output = generate_notes(
        model, network_input, pitchnames, n_vocab, device
    )
    create_midi(prediction_output, output_file="generated_nottingham_pytorch.mid")
    print(
        "> MIDI generation completed. Output saved to 'generated_nottingham_pytorch.mid'."
    )


if __name__ == "__main__":
    print("Starting MIDI Generation Example with PyTorch...")
    main()
