import gradio as gr
import music21
import tempfile
import os
import glob

# MuseScore 실행 파일 경로를 실제 경로로 바꿔주세요!
music21.environment.set('musescoreDirectPNGPath', r'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')

def midi_to_sheet(midi_bytes):
    # MIDI 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi:
        temp_midi.write(midi_bytes)
        temp_midi_path = temp_midi.name

    # music21로 MIDI 파싱
    score = music21.converter.parse(temp_midi_path)
    #print(">Score : ",score)

    # 프로그램 실행 디렉토리에 악보 이미지를 저장
    output_img_base = os.path.join(os.getcwd(), 'output_score.png')
    score.write('musicxml.png', fp=output_img_base)

    # 가장 최근에 생성된 output_score*.png 파일 찾기
    candidates = glob.glob(os.path.join(os.getcwd(), 'output_score*.png'))
    latest_img = max(candidates, key=os.path.getctime)
    print("> latest_img : ", latest_img)
    return latest_img

iface = gr.Interface(
    fn=midi_to_sheet,
    inputs=gr.File(type="binary", label="MIDI 파일 업로드"),
    outputs=gr.Image(type="filepath", label="오선지 악보 이미지"),
    title="MIDI 오선지 악보 변환기",
    description="MIDI 파일을 업로드하면 오선지에 음표가 나오는 악보 이미지로 변환합니다. MuseScore가 설치되어 있어야 합니다."
)

iface.launch()
