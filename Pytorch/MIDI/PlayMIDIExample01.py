import gradio as gr
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from matplotlib import rc

# 윈도우: Malgun Gothic, 맥: AppleGothic, 리눅스: NanumGothic 권장
import platform
if platform.system() == 'Windows':
    font_name = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    font_name = 'AppleGothic'
else:
    font_name = 'NanumGothic'
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지


def midi_to_piano_roll_img(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append([note.start, note.end, note.pitch])
    fig, ax = plt.subplots(figsize=(12, 5))
    if notes:
        notes = np.array(notes)
        for s, e, p in notes:
            ax.plot([s, e], [p, p], linewidth=8, solid_capstyle='butt')
        ax.set_ylim(20, 108)
    else:
        ax.text(0.5, 0.5, "No notes found in MIDI.", ha='center', va='center', fontsize=12)
        ax.set_axis_off()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MIDI Pitch')
    ax.set_title('MIDI Note Roll')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img

def midi_to_staff_img(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.pitch))
    if not notes:
        fig, ax = plt.subplots(figsize=(8,2))
        ax.text(0.5, 0.5, "No notes found in MIDI.", ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    min_pitch = 60  # C4
    max_pitch = 84  # C6

    notes = sorted(notes, key=lambda x: x[0])
    xs = [n[0] for n in notes]
    x_min = min(xs)
    x_max = max(xs)

    fig, ax = plt.subplots(figsize=(max(8, len(notes)//2), 4))
    staff_y = [0, 1, 2, 3, 4]
    staff_gap = 0.5
    for i in staff_y:
        ax.plot([0, 1], [i*staff_gap, i*staff_gap], color='black', lw=1, zorder=1)

    def pitch_to_y(pitch):
        return (pitch - min_pitch) * (staff_gap/2)
    def time_to_x(t):
        return 0.05 + 0.9*(t - x_min) / (x_max - x_min + 1e-6)
    for t, p in notes:
        if p < min_pitch or p > max_pitch:
            continue
        x = time_to_x(t)
        y = pitch_to_y(p)
        note = plt.Circle((x, y), 0.09, color='black', fill=False, lw=2, zorder=2)
        ax.add_patch(note)
        ax.plot([x+0.08, x+0.08], [y, y+0.7], color='black', lw=2, zorder=2)
    ax.set_ylim(-0.5, staff_gap*4+1.5)
    ax.set_xlim(0, 1.01)
    ax.set_axis_off()
    ax.set_title("오선지 위의 음표(간단 표기)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def process_midi(midi_file):
    if midi_file is None:
        return None, None
    piano_roll_img = midi_to_piano_roll_img(midi_file.name)
    staff_img = midi_to_staff_img(midi_file.name)
    return piano_roll_img, staff_img

with gr.Blocks() as demo:
    gr.Markdown("# MIDI 파일 음표 표시기")
    gr.Markdown(
        "MIDI 파일을 업로드하면<br>① 피아노롤<br>② 오선지 위 음표(간단 표기)<br>두 가지로 시각화합니다."
    )
    midi_input = gr.File(label="MIDI 파일 업로드", file_types=[".mid", ".midi"])
    with gr.Row():
        output_image1 = gr.Image(type="pil", label="피아노롤")
        output_image2 = gr.Image(type="pil", label="오선지 음표")
    gr.Button("변환 및 표시").click(
        process_midi,
        inputs=midi_input,
        outputs=[output_image1, output_image2]
    )

if __name__ == "__main__":
    demo.launch()