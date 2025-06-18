import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import miditoolkit


def initialize_model():
    model_name = "m-a-p/MIDI-LM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def generate_midi_sequence(model, tokenizer, prompt=""):
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate continuation tokens
    output = model.generate(
        input_ids,
        max_length=256,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        no_repeat_ngram_size=2,
    )

    # Decode tokens back to events
    generated_sequence = tokenizer.decode(output[0])
    return generated_sequence


# Convert to MIDI using miditoolkit
def sequence_to_midi(sequence, output_path="output.mid"):
    # Simple example: convert tokenized note numbers to MIDI notes
    midi_obj = miditoolkit.midi.parser.MidiFile()
    instrument = miditoolkit.Instrument(program=0, is_drum=False, name="Piano")
    time = 0
    for token in sequence.split():
        try:
            note = int(token)
            # Add a note with duration 480 (quarter note), simplistic
            midi_note = miditoolkit.Note(
                velocity=64, pitch=note, start=time, end=time + 480
            )
            instrument.notes.append(midi_note)
            time += 480
        except ValueError:
            continue
    midi_obj.instruments.append(instrument)
    midi_obj.dump(output_path)


def main():
    model, tokenizer = initialize_model()
    # Prompt (can be empty or a few notes in MIDI-LM token format)
    prompt = ""  # or e.g., "<start> 60 62 64 65 67 <end>"
    generated_sequence = generate_midi_sequence(model, tokenizer, prompt)
    sequence_to_midi(generated_sequence, "output.mid")
    print("MIDI file generated: output.mid")


if __name__ == "__main__":
    print("Generating MIDI with a pre-trained model...")
    main()
