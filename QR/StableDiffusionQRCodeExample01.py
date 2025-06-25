import torch
import qrcode
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import time

# --- 1. 모델 로딩 (스크립트 시작 시 한 번만 실행) ---
print("AI 모델을 전역으로 로딩합니다. 스크립트 시작 시 시간이 다소 걸릴 수 있습니다...")
start_time = time.time()

# ControlNet 및 Stable Diffusion 모델 로딩
# GPU가 사용 가능한지 확인하고 device 설정
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32 # CPU에서는 float32 사용

try:
    controlnet = ControlNetModel.from_pretrained(
        "DionTimmer/controlnet_qrcode-control_v1p_sd15",
        torch_dtype=torch_dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    
    # 스케줄러 설정
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # GPU 사용 시 메모리 최적화 (CPU에서는 비활성화)
    if device == "cuda":
        pipe.enable_model_cpu_offload()

    print(f"모델 로딩 완료! (소요 시간: {time.time() - start_time:.2f}초)")
    MODEL_LOADED = True
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    print("GPU 메모리 부족 또는 인터넷 연결 문제일 수 있습니다.")
    MODEL_LOADED = False

# --- 2. 이미지 생성 함수 (Gradio가 호출) ---
def create_artistic_qr_code(
    qr_data: str,
    prompt: str,
    negative_prompt: str,
    controlnet_conditioning_scale: float,
    seed: int
):
    """
    사용자 입력을 받아 예술적 QR 코드를 생성하고 PIL Image 객체와 상태 메시지를 반환합니다.
    """
    if not MODEL_LOADED:
        raise gr.Error("AI 모델이 로드되지 않았습니다. 프로그램을 재시작하거나 오류를 확인해주세요.")

    if not qr_data.strip():
        raise gr.Error("QR 코드에 담을 내용(URL 등)을 입력해주세요.")

    if not prompt.strip():
        raise gr.Error("이미지 프롬프트를 입력해주세요.")

    yield None, "1/3: 기본 QR 코드 생성 중..."
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    
    # 재현성을 위한 시드 고정
    # -1을 시드로 입력하면 무작위 시드 사용
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.manual_seed(seed)

    yield None, f"2/3: 이미지 생성 중... (시드: {seed})"

    # 파이프라인 실행
    generated_image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=qr_image,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    ).images[0]

    yield generated_image, "3/3: 생성 완료! 스마트폰으로 스캔을 테스트해보세요."


# --- 3. Gradio 인터페이스 설정 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 AI 예술적 QR 코드 생성기")
    gr.Markdown("QR 코드에 담을 내용과 만들고 싶은 이미지에 대한 설명을 입력하면, AI가 스캔 가능한 예술 작품을 만들어줍니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            qr_data_input = gr.Textbox(
                label="QR 코드에 담을 내용 (URL 등)", 
                placeholder="https://gemini.google.com/"
            )
            prompt_input = gr.Textbox(
                label="이미지 프롬프트 (영문)", 
                placeholder="A medieval castle on a mountain, dramatic sky, fantasy, detailed oil painting", 
                lines=3
            )
            negative_prompt_input = gr.Textbox(
                label="제외할 요소 (Negative Prompt)", 
                placeholder="ugly, disfigured, low quality, blurry, nsfw",
                lines=2
            )
            scale_slider = gr.Slider(
                minimum=0.8, maximum=2.5, step=0.05, value=1.4,
                label="QR 코드 패턴 강도 (ControlNet Scale)",
                info="높을수록 QR 패턴이 선명해지나, 예술성이 낮아집니다. (권장: 1.3 ~ 1.6)"
            )
            seed_input = gr.Number(
                label="시드 (Seed)", value=-1,
                info="같은 시드는 같은 결과를 보장합니다. -1을 입력하면 무작위로 생성됩니다."
            )
            submit_button = gr.Button("QR 코드 생성하기", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="결과 이미지")
            status_textbox = gr.Textbox(label="진행 상태", interactive=False)
            gr.Markdown("### 💡 사용 팁\n"
                        "1. 생성이 완료되면 우측 상단 다운로드 버튼으로 이미지를 저장할 수 있습니다.\n"
                        "2. **'QR 코드 패턴 강도'** 슬라이더를 조절하며 최적의 결과물을 찾아보세요.\n"
                        "3. **가장 중요한 점: 생성된 QR 코드는 반드시 스마트폰으로 스캔 테스트를 진행하세요!**")

    # 버튼 클릭 시 함수 실행
    submit_button.click(
        fn=create_artistic_qr_code,
        inputs=[
            qr_data_input,
            prompt_input,
            negative_prompt_input,
            scale_slider,
            seed_input
        ],
        outputs=[output_image, status_textbox]
    )


if __name__ == '__main__':
    if MODEL_LOADED:
        demo.launch(share=True) # share=True로 설정 시 외부 접속용 public URL 생성
    else:
        print("\n모델 로딩에 실패하여 Gradio 앱을 시작할 수 없습니다.")