import torch
import platform
from diffusers import FluxPipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr


#DEFAULT_PROMPT = "The image is a vertical, ultra-high-quality beautiful young skinny korean girl in a modern kitchen. The style blends sharp  character design with hyper-realistic textures, particularly in the rendering of skin and surfaces. The Character: Appearance: She has a fair complexion with a soft pink blush and striking light blue eyes. Her expression is calm and confident. Hair: Her hair is long, wavy, and a pale black color. It is tied back into a high, voluminous messy ponytail with a black scrunchie, while loose strands frame her face. Skin: Her skin is rendered with an extremely high-gloss, wet or oily finish, creating brilliant specular highlights on her shoulders, midriff, and legs. Pose: She is standing in front of a kitchen counter, leaning her left hand on the marble surface while her right hand is raised to her hair. Attire: Top: A form-fitting, sleeveless white high-neck crop top that emphasizes her silhouette. Bottom: Simple, high-cut teal or turquoise bikini-style bottoms. Setting & Background: Location: A clean, modern, and well-lit kitchen. Furniture: She stands before white paneled cabinets with silver hardware. Above the counter is an open shelving unit containing various glass jars of ingredients, spices, and a small wooden bowl. Details: The countertop is a white polished marble. To her right, the wall is covered in white square tiles with blue grout. Common kitchen items like wooden spatulas in a canister and a silver kettle are visible. Atmosphere: The setting is bright, airy, and organized, creating a professional yet domestic feel. Lighting: Effect: The scene is illuminated by bright, even indoor lighting. This creates high-contrast highlights on the character's glossy skin and the reflective surfaces of the marble and tiling. The shadows are soft and realistic. perfect anatomy, detailed background, intricate details, 8k, high resolution, photorealistic, ultra-detailed, sharp focus, studio lighting."

DEFAULT_PROMPT = "The image is a high-quality realistic photograph of a beautiful young Korean girl. The composition is a wide medium-shot with an airy lifestyle aesthetic, capturing a serene and pensive moment indoors. The Character: Appearance: She has a fair, radiant complexion and a calm, direct gaze. Her expression is soft and youthful. Hair: Her hair is a warm chestnut brown, styled in a sleek, straight chin-length bob with full bangs framing her forehead. Pose: She is reclining gracefully on a sofa, with her legs extended along the cushions. Her right arm rests on the back of the sofa, while her left hand is tucked near her waist. Accessories: She wears delicate gold dangling earrings. Attire: Outfit: She wears a form-fitting, strapless white corset-style bodysuit. Details: The corset features intricate white-on-white floral embroidery and a scalloped lace trim at the top. The fabric has a subtle silken sheen. Setting & Background: Location: A bright, minimalist indoor room with a clean and peaceful atmosphere. Furniture: A modern, low-profile white tufted fabric sofa with soft cushions. Backdrop: Large, airy white translucent curtains hang in the background, diffusing the light. Floor: The room has natural, light-toned wooden floorboards. Lighting: Effect: The scene is characterized by intense, high-key natural lighting. The light is soft and diffused, creating a dreamy or hazy effect that softens the edges of the subject and the furniture. There is a subtle lens flare and a bright glow coming from the window side (right), emphasizing an ethereal and pristine mood. Perfect anatomy, detailed background, intricate details, 8k, high resolution, photorealistic, ultra-detailed, sharp focus, studio lighting."

#DEFAULT_PROMPT = "A stunning futuristic depiction of Ciri cute korean girl astronaut aboard an advanced spacecraft, wearing a sleek, form-fitting, and partially revealing neon pink holographic spacesuit. The suit is designed with a futuristic aesthetic, blending cutting- edge technology and elegance. The material is a smooth, see-through fabric with a subtle sheen, accentuating her athletic physique. The outfit features deep cutouts in the chest and side areas, exposing skin in a way that highlights her form while maintaining a futuristic and functional look. The high collar and structured panels give it a space-age appearance, while the open design emphasizes freedom of movement and aesthetic appeal.Her left arm is partially wrapped in a textured band, resembling a combination of fabric and lightweight armor, adding detail and a sense of functionality. The exposed areas of her skin contrast with the polished white fabric, giving the outfit a bold and daring look, yet it remains cohesive within the context of the sci-fi setting. Her confident pose, with both arms resting on her head, enhances her commanding and elegant presence. The background is the interior of a high-tech spacecraft, with glowing control panels, intricate machinery, and large observation windows offering a breathtaking view of a nebula galaxy against the starry expanse of space. The lighting is dynamic, with soft highlights reflecting off her suit and metallic surfaces, adding depth and realism to the scene. The overall atmosphere is a blend of elegance, advanced technology, and the thrill of space exploration, evoking themes of humanity's push into the cosmos and the resilience of the human spirit."


def get_device_and_dtype():
    """Detect the best available device and appropriate data type."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


# Global variables
DEVICE, DTYPE = get_device_and_dtype()
pipe = None
interface = None


def get_available_devices():
    """Return list of available device choices.
    - CUDA + CPU: both selectable
    - MPS only (no CUDA): MPS only
    - No GPU: CPU only
    """
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
        devices.append("cpu")
    elif torch.backends.mps.is_available():
        devices.append("mps")
    else:
        devices.append("cpu")
    return devices


def print_hardware_info():
    """Print detailed hardware specifications."""
    print("=" * 60)
    print("하드웨어 사양")
    print("=" * 60)

    # OS 정보
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"OS 버전: {platform.version()}")
    print(f"아키텍처: {platform.machine()}")

    # Python 정보
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # CPU 정보
    print("-" * 60)
    print("CPU 정보")
    print("-" * 60)
    print(f"프로세서: {platform.processor()}")
    print(f"물리 코어: {psutil.cpu_count(logical=False)}")
    print(f"논리 코어: {psutil.cpu_count(logical=True)}")

    # 메모리 정보
    mem = psutil.virtual_memory()
    print("-" * 60)
    print("메모리 정보")
    print("-" * 60)
    print(f"총 RAM: {mem.total / (1024**3):.1f} GB")
    print(f"사용 가능: {mem.available / (1024**3):.1f} GB")
    print(f"사용률: {mem.percent}%")

    # GPU 정보
    print("-" * 60)
    print("GPU 정보")
    print("-" * 60)
    if torch.cuda.is_available():
        print("CUDA 사용 가능: 예")
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - VRAM: {props.total_memory / (1024**3):.1f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - 멀티프로세서: {props.multi_processor_count}")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon) 사용 가능: 예")
        print(f"디바이스: {platform.processor()}")
    else:
        print("GPU 사용 불가 (CPU 모드)")

    print("=" * 60)


def cleanup():
    """Release all resources and clear memory."""
    global pipe, interface
    print("\n리소스 정리 중...")

    if interface is not None:
        try:
            interface.close()
            print("Gradio 서버 종료됨")
        except Exception:
            pass
        interface = None

    if pipe is not None:
        del pipe
        pipe = None
        print("모델 해제됨")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA 캐시 정리됨")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("MPS 캐시 정리됨")

    gc.collect()
    print("메모리 정리 완료")


def signal_handler(_sig, _frame):
    """Handle keyboard interrupt signal."""
    print("\n\n키보드 인터럽트 감지됨 (Ctrl+C)")
    cleanup()
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def load_model(device_name=None):
    """Load and initialize the Flux model with optimizations."""
    global pipe, DEVICE, DTYPE

    if device_name is not None:
        DEVICE = device_name
        DTYPE = torch.bfloat16 if device_name in ("cuda", "mps") else torch.float32

    # Release previous model if loaded
    if pipe is not None:
        print("기존 모델 해제 중...")
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"모델 로딩 중... (Device: {DEVICE}, dtype: {DTYPE})")
    print("T5-XXL 텍스트 인코더만 사용합니다. (CLIP 비활성화)")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder=None,
        tokenizer=None,
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)

    # Enable memory optimizations based on device
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print(
            "메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CUDA)"
        )
    elif DEVICE == "cpu":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print(
            "메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CPU)"
        )
    elif DEVICE == "mps":
        # MPS doesn't support cpu_offload well
        print("No memory optimizations applied for MPS device.")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    strength,
    max_sequence_length,
    progress=gr.Progress(track_tqdm=True),
):
    global pipe

    if pipe is None:
        return (
            None,
            "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요.",
        )

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="프롬프트 인코딩 중...")

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        # Encode prompt using T5 only (CLIP is disabled)
        text_inputs = pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=int(max_sequence_length),
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds = pipe.text_encoder_2(
                text_inputs["input_ids"].to(DEVICE),
                output_hidden_states=False,
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=DTYPE)

        # Zero pooled embeddings (normally from CLIP, not needed with T5-only)
        pooled_prompt_embeds = torch.zeros(
            1, 768, dtype=DTYPE, device=prompt_embeds.device
        )

        progress(0.05, desc="추론 시작...")

        # Callback to report each inference step to Gradio progress bar
        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            # Map step progress to 0.05 ~ 0.90 range
            progress_val = 0.05 + ratio * 0.85
            progress(progress_val, desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)")
            return callback_kwargs

        # Build pipeline kwargs with pre-computed embeddings
        pipe_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": step_callback,
        }

        # Run the pipeline
        image = pipe(**pipe_kwargs).images[0]

        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}_{timestamp}_{DEVICE.upper()}_{width}x{height}_gs{guidance_scale}_step{steps}_seed{int(seed)}_str{strength}_msl{int(max_sequence_length)}.png"

        print(f"이미지가 저장되었습니다 : {filename}")
        image.save(filename)

        progress(1.0, desc="완료!")
        return image, f"✓ 완료! ({elapsed:.1f}초) 저장됨: {filename}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    # Print hardware specifications
    print_hardware_info()

    # Auto-load model on startup with detected device
    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    # Create Gradio interface
    with gr.Blocks(title="Flux.1-dev Text-to-Image Generator") as interface:
        gr.Markdown("# Flux.1-dev Text-to-Image Generator")
        gr.Markdown(
            f"AI를 사용하여 텍스트에서 이미지를 생성합니다. (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                device_selector = gr.Radio(
                    label="디바이스 선택",
                    choices=get_available_devices(),
                    value=DEVICE,
                    info="모델을 실행할 디바이스를 선택하세요.",
                )
                load_model_btn = gr.Button("모델 로드", variant="secondary")
                device_status = gr.Textbox(
                    label="모델 상태",
                    value=(
                        f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"
                        if pipe is not None
                        else "모델이 로드되지 않았습니다. 디바이스를 선택하고 '모델 로드' 버튼을 눌러주세요."
                    ),
                    interactive=False,
                )

                # Input parameters
                prompt = gr.Textbox(
                    label="프롬프트",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    placeholder="이미지에 대한 설명을 입력하세요 (77단어 이하 권장)",
                    info="생성하고 싶은 이미지에 대한 텍스트 설명입니다. 자세할수록 좋습니다. 예: '여자, 미소, 해변, 빨간 비키니'",
                )
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=768,
                        info="생성할 이미지의 너비를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1536,
                        info="생성할 이미지의 높이를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=4.0,
                        info="모델이 프롬프트를 얼마나 따를지 제어합니다. 낮을수록 창의적, 높을수록 정확합니다. 권장: 4-15",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=28,
                        info="이미지 생성 과정의 단계 수입니다. 높을수록 품질이 좋지만 시간이 더 걸립니다. 권장: 20-28",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        value=42,
                        precision=0,
                        info="난수 생성의 시작점입니다. 같은 시드를 사용하면 같은 결과를 얻습니다.",
                    )
                    strength = gr.Slider(
                        label="강도",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.8,
                        info="생성 모델의 강도를 제어합니다. 낮을수록 다양한 결과, 높을수록 일관성 있는 결과입니다.",
                    )

                with gr.Row():
                    max_sequence_length = gr.Slider(
                        label="최대 시퀀스 길이",
                        minimum=64,
                        maximum=512,
                        step=64,
                        value=512,
                        info="텍스트 인코더의 최대 시퀀스 길이입니다. 긴 프롬프트를 사용할 경우 높은 값이 필요합니다.",
                    )

                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="생성된 이미지", height=800)
                output_message = gr.Textbox(label="상태", interactive=False)

        # Load model when button is clicked
        load_model_btn.click(
            fn=load_model,
            inputs=[device_selector],
            outputs=[device_status],
        )

        # Auto-load model when device is changed
        device_selector.change(
            fn=load_model,
            inputs=[device_selector],
            outputs=[device_status],
        )

        # Connect the generate button to the function
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                strength,
                max_sequence_length,
            ],
            outputs=[output_image, output_message],
        )

    # Launch the interface
    interface.launch(inbrowser=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
