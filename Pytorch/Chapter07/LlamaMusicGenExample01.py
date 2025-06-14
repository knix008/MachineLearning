import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Install mido and python-rtmidi if needed:
# pip install mido python-rtmidi

from huggingface_hub import login
# Login to Hugging Face Hub if needed
login(token="your_huggingface_token_here")  # Replace with your actual token

import mido

def parse_generated_midi_events(text):
    """
    Parses LLM-generated MIDI event lines like:
    note_on 60 64 0.5
    note_off 60 0.5
    Returns a list of mido.Message objects.
    """
    events = []
    time_acc = 0
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "note_on" and len(parts) == 4:
            _, note, velocity, duration = parts
            events.append({
                "type": "note_on",
                "note": int(note),
                "velocity": int(velocity),
                "time": time_acc
            })
            time_acc = float(duration)
        elif parts[0] == "note_off" and len(parts) == 3:
            _, note, duration = parts
            events.append({
                "type": "note_off",
                "note": int(note),
                "velocity": 0,
                "time": time_acc
            })
            time_acc = float(duration)
    return events

def events_to_midi(events, filename="generated_llama_music.mid", tempo=500000):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    last_tick = 0
    ticks_per_beat = mid.ticks_per_beat
    for event in events:
        msg_time = mido.second2tick(event['time'], ticks_per_beat, tempo)
        if event['type'] == 'note_on':
            track.append(mido.Message('note_on', note=event['note'], velocity=event['velocity'], time=int(msg_time)))
        elif event['type'] == 'note_off':
            track.append(mido.Message('note_off', note=event['note'], velocity=0, time=int(msg_time)))
    
    mid.save(filename)
    print(f"MIDI file saved as {filename}")

# Step 1: Load Llama LLM model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Use your preferred Llama model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Step 2: Prompt to generate MIDI event style music
prompt = (
    "Generate a simple melody as a sequence of MIDI events, one per line, with the format:\n"
    "note_on <note_number> <velocity> <duration_seconds>\n"
    "note_off <note_number> <duration_seconds>\n"
    "Example:\n"
    "note_on 60 64 0.5\n"
    "note_off 60 0.5\n"
    "note_on 62 64 0.5\n"
    "note_off 62 0.5\n"
    "Melody:\n"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Step 3: Generate event sequence
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=256,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated MIDI events:\n", generated)

# Step 4: Extract the MIDI events after "Melody:"
if "Melody:" in generated:
    midi_events_text = generated.split("Melody:")[1].strip()
else:
    midi_events_text = generated

events = parse_generated_midi_events(midi_events_text)
events_to_midi(events, filename="generated_llama_music.mid")