import gradio as gr
import time
from datetime import datetime, timedelta
import threading

# 전역 변수
meeting_start_time = None
timer_end_time = None
is_meeting_active = False
timer_thread = None
stop_timer_flag = False


def format_time_elapsed(seconds):
    """경과 시간을 HH:MM:SS 형식으로 포맷"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def start_meeting():
    """회의 시작"""
    global meeting_start_time, is_meeting_active
    meeting_start_time = time.time()
    is_meeting_active = True
    return (
        gr.update(value="회의 진행 중...", visible=True),
        gr.update(value="00:00:00"),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value=""),
    )


def stop_meeting():
    """회의 종료"""
    global meeting_start_time, is_meeting_active, timer_end_time, stop_timer_flag
    is_meeting_active = False
    meeting_start_time = None
    timer_end_time = None
    stop_timer_flag = True
    return (
        gr.update(value="회의가 종료되었습니다.", visible=True),
        gr.update(value="00:00:00"),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=""),
    )


def get_elapsed_time():
    """현재 경과 시간 반환"""
    if is_meeting_active and meeting_start_time:
        elapsed = time.time() - meeting_start_time
        return format_time_elapsed(elapsed)
    return "00:00:00"


def set_timer(minutes):
    """타이머 설정"""
    global timer_end_time, stop_timer_flag, timer_thread
    
    if not is_meeting_active:
        return "⚠️ 먼저 회의를 시작하세요."
    
    if minutes <= 0:
        return "⚠️ 0보다 큰 시간을 입력하세요."
    
    timer_end_time = time.time() + (minutes * 60)
    stop_timer_flag = False
    
    # 타이머 스레드 시작
    if timer_thread and timer_thread.is_alive():
        stop_timer_flag = True
        timer_thread.join()
    
    timer_thread = threading.Thread(target=timer_worker, args=(minutes,))
    timer_thread.daemon = True
    timer_thread.start()
    
    return f"⏰ 타이머 설정됨: {int(minutes)}분 후 알림"


def timer_worker(minutes):
    """타이머 백그라운드 작업"""
    global stop_timer_flag
    end_time = time.time() + (minutes * 60)
    
    while time.time() < end_time and not stop_timer_flag:
        time.sleep(1)
    
    if not stop_timer_flag:
        # 타이머 종료 알림 (콘솔)
        print(f"\n{'='*50}")
        print(f"⏰ 타이머 종료! {int(minutes)}분이 경과했습니다.")
        print(f"{'='*50}\n")


def clear_timer():
    """타이머 초기화"""
    global timer_end_time, stop_timer_flag
    timer_end_time = None
    stop_timer_flag = True
    return "타이머가 초기화되었습니다."


def get_timer_status():
    """타이머 남은 시간 반환"""
    if timer_end_time and not stop_timer_flag:
        remaining = timer_end_time - time.time()
        if remaining > 0:
            return f"⏰ 남은 시간: {format_time_elapsed(remaining)}"
        else:
            return "🔔 타이머 종료! 설정한 시간이 지났습니다."
    return ""


def update_display():
    """실시간 업데이트 (경과 시간 및 타이머 상태)"""
    elapsed = get_elapsed_time()
    timer_status = get_timer_status()
    return elapsed, timer_status


# Gradio UI
with gr.Blocks(title="회의 타이머") as app:
    gr.Markdown("# 📊 회의 타이머")
    gr.Markdown("회의 시작부터 경과 시간을 추적하고, 타이머를 설정하여 알림을 받을 수 있습니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 회의 제어")
            start_btn = gr.Button("🚀 회의 시작", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ 회의 종료", variant="stop", size="lg", interactive=False)
            
            status_text = gr.Textbox(
                label="상태",
                value="회의를 시작하세요.",
                interactive=False,
                visible=True,
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 경과 시간")
            elapsed_display = gr.Textbox(
                label="회의 진행 시간",
                value="00:00:00",
                interactive=False,
                elem_classes="elapsed-time-display",
            )
            
            gr.Markdown("---")
            
            gr.Markdown("### ⏰ 타이머 설정")
            timer_minutes = gr.Number(
                label="알림 시간 (분)",
                value=30,
                precision=0,
                minimum=1,
                maximum=480,
                info="몇 분 후에 알림을 받을지 설정하세요.",
            )
            
            with gr.Row():
                set_timer_btn = gr.Button("⏰ 타이머 시작", variant="secondary", interactive=False)
                clear_timer_btn = gr.Button("🗑️ 타이머 초기화", variant="secondary", interactive=False)
            
            timer_status_text = gr.Textbox(
                label="타이머 상태",
                value="",
                interactive=False,
            )
    
    # 실시간 업데이트를 위한 숨겨진 컴포넌트
    update_interval = gr.Number(value=0, visible=False)
    
    # 이벤트 핸들러
    start_btn.click(
        fn=start_meeting,
        outputs=[status_text, elapsed_display, start_btn, stop_btn, set_timer_btn, timer_status_text],
    )
    
    stop_btn.click(
        fn=stop_meeting,
        outputs=[status_text, elapsed_display, start_btn, stop_btn, set_timer_btn, timer_status_text],
    )
    
    set_timer_btn.click(
        fn=set_timer,
        inputs=[timer_minutes],
        outputs=[timer_status_text],
    )
    
    clear_timer_btn.click(
        fn=clear_timer,
        outputs=[timer_status_text],
    )
    
    # 1초마다 자동 업데이트
    app.load(
        fn=update_display,
        outputs=[elapsed_display, timer_status_text],
        every=1,
    )
    
    # CSS 스타일링
    app.css = """
    .elapsed-time-display textarea {
        font-size: 48px !important;
        font-weight: bold !important;
        text-align: center !important;
        font-family: 'Courier New', monospace !important;
        color: #2563eb !important;
    }
    """


if __name__ == "__main__":
    print("회의 타이머 시작...")
    print("브라우저에서 인터페이스가 열립니다.")
    app.launch(inbrowser=True)
