import gradio as gr
import pretty_midi
import matplotlib.pyplot as plt
import tempfile
import os

def midi_to_pianoroll(midi_file):
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi:
        temp_midi.write(midi_file)  # .read() 제거
        temp_midi_path = temp_midi.name

    midi_data = pretty_midi.PrettyMIDI(temp_midi_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            piano_roll = instrument.get_piano_roll(fs=100)
            ax.imshow(piano_roll, aspect='auto', origin='lower', cmap='gray_r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pitch')
    plt.tight_layout()
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_img.name)
    plt.close(fig)
    os.remove(temp_midi_path)
    return temp_img.name

iface = gr.Interface(
    fn=midi_to_pianoroll,
    inputs=gr.File(type="binary", label="MIDI 파일 업로드"),
    outputs=gr.Image(type="filepath", label="피아노 롤 시각화"),
    title="MIDI 피아노 롤 시각화",
    description="MIDI 파일을 업로드하면 피아노 롤로 시각화합니다."
)

iface.launch()
