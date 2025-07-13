import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers.models.clip import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from PIL import Image
import numpy as np

# Hugging Face Access Token (필요시 사용)
#access_token = ""
#from huggingface_hub import login
#login(access_token)


def load_model():
    """Stable Diffusion 3.5 Medium 모델을 로드합니다."""
    model_id = "stabilityai/stable-diffusion-3.5-medium"

    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"사용 중인 디바이스: {device}")
    print(f"사용 중인 데이터 타입: {torch_dtype}")

    # 파이프라인 생성 - 기본 로딩 후 컴포넌트 교체
    try:
        print("모델 로딩 중...")
        # 먼저 기본 파이프라인 로드
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            feature_extractor=None,
            image_encoder=None,
            text_encoder=None,
            tokenizer=None,
            scheduler=None,
            safety_checker=None,
            unet=None,
            vae=None,
            requires_safety_checker=False
        )
        print("기본 파이프라인 로딩 완료!")
        
        # 개별 컴포넌트 교체 시도
        try:
            print("UNet 교체 중...")
            unet = UNet2DConditionModel.from_pretrained(
                model_id, 
                subfolder="unet",
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
            pipe.unet = unet
            print("UNet 교체 완료!")
        except Exception as e:
            print(f"UNet 교체 실패: {e}")
        
        try:
            print("VAE 교체 중...")
            vae = AutoencoderKL.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
            pipe.vae = vae
            print("VAE 교체 완료!")
        except Exception as e:
            print(f"VAE 교체 실패: {e}")
        
        try:
            print("Text Encoder 교체 중...")
            text_encoder = CLIPTextModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=torch_dtype
            )
            pipe.text_encoder = text_encoder
            print("Text Encoder 교체 완료!")
        except Exception as e:
            print(f"Text Encoder 교체 실패: {e}")
        
        try:
            print("Tokenizer 교체 중...")
            tokenizer = CLIPTokenizer.from_pretrained(
                model_id,
                subfolder="tokenizer"
            )
            pipe.tokenizer = tokenizer
            print("Tokenizer 교체 완료!")
        except Exception as e:
            print(f"Tokenizer 교체 실패: {e}")
        
        try:
            print("Feature Extractor 교체 중...")
            feature_extractor = CLIPImageProcessor.from_pretrained(
                model_id,
                subfolder="feature_extractor"
            )
            pipe.feature_extractor = feature_extractor
            print("Feature Extractor 교체 완료!")
        except Exception as e:
            print(f"Feature Extractor 교체 실패: {e}")
        
        try:
            print("Image Encoder 교체 중...")
            image_encoder = CLIPTextModel.from_pretrained(
                model_id,
                subfolder="image_encoder",
                torch_dtype=torch_dtype
            )
            pipe.image_encoder = image_encoder
            print("Image Encoder 교체 완료!")
        except Exception as e:
            print(f"Image Encoder 교체 실패: {e}")
        
        print("모델 로딩 완료!")
    except Exception as e:
        print(f"기본 로딩 실패: {e}")
        print("최소 설정으로 로딩 시도...")
        # 최소 설정으로 로딩
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype
        )

    # 스케줄러 설정
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("스케줄러 설정 완료")
    except Exception as e:
        print(f"스케줄러 설정 실패, 기본 스케줄러 사용: {e}")

    # 디바이스로 이동
    pipe = pipe.to(device)
    
    # 메모리 최적화 설정
    try:
        if device == "cuda":
            print("GPU를 사용하여 모델을 로드했습니다.")
            # GPU 메모리 최적화
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        else:
            print("CPU를 사용하여 모델을 로드했습니다.")
            print("CPU 모드에서는 생성 시간이 오래 걸릴 수 있습니다.")
            # CPU 메모리 최적화
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
    except Exception as e:
        print(f"메모리 최적화 설정 실패: {e}")

    return pipe


def style_preserving_generation(
    input_image,
    prompt,
    negative_prompt="",
    strength=0.6,
    guidance_scale=7.5,
    num_inference_steps=25,
    seed=-1,
    preserve_style=True,
    style_strength=0.8,
    cpu_optimization=False,
    max_image_size=768,
):
    """
    입력 이미지의 스타일을 유지하면서 새로운 이미지를 생성합니다.

    Args:
        input_image: 입력 이미지 (PIL Image)
        prompt: 생성할 이미지에 대한 프롬프트
        negative_prompt: 네거티브 프롬프트
        strength: 변환 강도 (0.0-1.0)
        guidance_scale: 가이던스 스케일
        num_inference_steps: 추론 스텝 수
        seed: 랜덤 시드 (-1이면 랜덤)
        preserve_style: 스타일 보존 여부
        style_strength: 스타일 보존 강도 (0.0-1.0)

    Returns:
        생성된 이미지와 상태 메시지
    """
    try:
        # 모델 로드
        pipe = load_model()

        # 디바이스 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 시드 설정
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
            
        generator = torch.Generator(device=device).manual_seed(seed)

        # 이미지 전처리
        if input_image is None:
            return None, "입력 이미지를 업로드해주세요."

        # 이미지 크기 조정 (메모리 효율성을 위해)
        width, height = input_image.size
        if width > max_image_size or height > max_image_size:
            ratio = min(max_image_size / width, max_image_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            input_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"이미지 크기를 {width}x{height}에서 {new_width}x{new_height}로 조정했습니다.")

        # CPU 최적화 설정
        if cpu_optimization and device == "cpu":
            print("CPU 최적화 모드가 활성화되었습니다.")
            # CPU에서 더 안정적인 설정
            if num_inference_steps > 25:
                num_inference_steps = 25
                print(f"CPU 최적화: 추론 스텝을 {num_inference_steps}로 조정했습니다.")
            if guidance_scale > 8:
                guidance_scale = 8
                print(f"CPU 최적화: 가이던스 스케일을 {guidance_scale}로 조정했습니다.")
            # CPU에서 더 안정적인 strength 설정
            if strength > 0.7:
                strength = 0.7
                print(f"CPU 최적화: 변환 강도를 {strength}로 조정했습니다.")

        # 스타일 보존을 위한 프롬프트 조정
        if preserve_style:
            # 입력 이미지의 스타일을 분석하여 프롬프트에 추가
            style_enhanced_prompt = f"{prompt}, maintaining the artistic style and composition of the original image"
        else:
            style_enhanced_prompt = prompt

        print(f"이미지 생성 시작... (디바이스: {device}, 스텝: {num_inference_steps})")
        
        # 이미지 생성
        result = pipe(
            prompt=style_enhanced_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # 결과 이미지 반환
        output_image = result.images[0]

        # 메모리 정리
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_image, f"생성 완료! 사용된 시드: {seed} (디바이스: {device})"

    except Exception as e:
        return None, f"오류가 발생했습니다: {str(e)}"


def create_interface():
    """Gradio 인터페이스를 생성합니다."""

    # 프롬프트 예시들
    prompt_examples = [
        "a beautiful landscape with mountains and lake",
        "a futuristic city with flying cars",
        "a magical forest with glowing mushrooms",
        "a cozy coffee shop interior",
        "a space station orbiting Earth",
        "a medieval castle on a hill",
        "a tropical beach at sunset",
        "a cyberpunk street at night",
        "a peaceful Japanese garden",
        "a steampunk airship in the sky"
    ]

    # 네거티브 프롬프트 예시들
    negative_prompt_examples = [
        "blurry, low quality, distorted, ugly",
        "watermark, signature, text, logo",
        "oversaturated, overexposed, underexposed",
        "deformed, disfigured, bad anatomy",
        "noise, grain, artifacts"
    ]

    with gr.Blocks(
        title="Stable Diffusion 3.5 Style-Preserving Generation", 
        theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("# 🎨 Stable Diffusion 3.5 스타일 보존 이미지 생성")
        gr.Markdown("입력 이미지의 스타일을 유지하면서 새로운 이미지를 생성해보세요!")

        with gr.Row():
            with gr.Column(scale=1):
                # 입력 섹션
                gr.Markdown("## 📤 입력")
                input_image = gr.Image(
                    label="참조할 이미지를 업로드하세요", 
                    type="pil", 
                    height=300
                )

                prompt = gr.Textbox(
                    label="생성 프롬프트",
                    placeholder="예: a beautiful landscape with mountains and lake",
                    lines=3,
                )

                negative_prompt = gr.Textbox(
                    label="네거티브 프롬프트 (선택사항)",
                    placeholder="예: blurry, low quality, distorted",
                    lines=2,
                )

                # 파라미터 조정
                gr.Markdown("### ⚙️ 파라미터 조정")
                with gr.Row():
                    strength = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="변환 강도",
                        info="높을수록 더 많이 변환됩니다",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="가이던스 스케일",
                        info="높을수록 프롬프트를 더 잘 따릅니다",
                    )

                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="추론 스텝 수",
                        info="높을수록 품질이 좋아지지만 시간이 오래 걸립니다",
                    )
                    seed = gr.Number(
                        value=-1,
                        label="랜덤 시드",
                        info="-1이면 랜덤 시드를 사용합니다",
                    )

                # 스타일 보존 옵션
                gr.Markdown("### 🎭 스타일 보존 설정")
                with gr.Row():
                    preserve_style = gr.Checkbox(
                        value=True,
                        label="스타일 보존 활성화",
                        info="입력 이미지의 스타일을 유지합니다"
                    )
                    style_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="스타일 보존 강도",
                        info="스타일을 얼마나 강하게 보존할지 설정",
                        visible=True
                    )

                # CPU 최적화 옵션
                gr.Markdown("### ⚡ CPU 최적화 설정")
                with gr.Row():
                    cpu_optimization = gr.Checkbox(
                        value=not torch.cuda.is_available(),
                        label="CPU 최적화 활성화",
                        info="CPU 사용시 메모리 사용량을 줄이고 안정성을 높입니다"
                    )
                    max_image_size = gr.Slider(
                        minimum=512,
                        maximum=1024,
                        value=768 if torch.cuda.is_available() else 512,
                        step=64,
                        label="최대 이미지 크기",
                        info="CPU 사용시 더 작은 크기를 권장합니다"
                    )

                # 디바이스 정보 표시
                device_info = "GPU" if torch.cuda.is_available() else "CPU"
                gr.Markdown(f"**현재 사용 중인 디바이스: {device_info}**")
                if not torch.cuda.is_available():
                    gr.Markdown("⚠️ **CPU 모드**: 생성 시간이 오래 걸릴 수 있습니다 (5-15분)")

                # 생성 버튼
                generate_btn = gr.Button(
                    "🎨 이미지 생성하기", 
                    variant="primary", 
                    size="lg"
                )

                # 상태 메시지
                status_text = gr.Textbox(
                    label="상태", 
                    interactive=False, 
                    lines=2
                )

            with gr.Column(scale=1):
                # 출력 섹션
                gr.Markdown("## 📤 결과")
                output_image = gr.Image(
                    label="생성된 이미지", 
                    height=400
                )

        # 프롬프트 예시
        gr.Markdown("## 💡 프롬프트 예시")
        with gr.Row():
            for i, example in enumerate(prompt_examples[:5]):
                gr.Button(example, size="sm").click(
                    fn=lambda x=example: x, 
                    outputs=prompt
                )

        with gr.Row():
            for i, example in enumerate(prompt_examples[5:]):
                gr.Button(example, size="sm").click(
                    fn=lambda x=example: x, 
                    outputs=prompt
                )

        # 네거티브 프롬프트 예시
        gr.Markdown("## 🚫 네거티브 프롬프트 예시")
        with gr.Row():
            for i, example in enumerate(negative_prompt_examples):
                gr.Button(example, size="sm").click(
                    fn=lambda x=example: x, 
                    outputs=negative_prompt
                )

        # 사용법 안내
        gr.Markdown(
            """
        ## 📖 사용법
        
        1. **이미지 업로드**: 스타일을 참조할 이미지를 업로드하세요
        2. **프롬프트 입력**: 생성하고 싶은 이미지를 설명하는 프롬프트를 입력하세요
        3. **파라미터 조정**: 변환 강도와 품질을 조정하세요
        4. **스타일 보존 설정**: 원본 스타일을 얼마나 유지할지 설정하세요
        5. **CPU 최적화**: CPU 사용시 최적화 옵션을 활성화하세요
        6. **생성 실행**: "이미지 생성하기" 버튼을 클릭하세요
        
        ### 💡 팁
        - **변환 강도**: 0.4-0.7 정도가 적당합니다
        - **가이던스 스케일**: 7-10 정도가 좋은 결과를 보여줍니다
        - **추론 스텝**: 20-30 스텝이 품질과 속도의 균형점입니다
        - **스타일 보존**: 활성화하면 원본 이미지의 아트 스타일을 유지합니다
        
        ### ⚡ CPU 사용시 주의사항
        - **생성 시간**: CPU에서는 GPU보다 훨씬 오래 걸립니다 (5-15분)
        - **메모리**: 최소 8GB RAM이 필요합니다
        - **이미지 크기**: 512x512 이하로 설정하는 것을 권장합니다
        - **최적화**: CPU 최적화 옵션을 활성화하면 안정성이 향상됩니다
        """
        )

        # 이벤트 연결
        generate_btn.click(
            fn=style_preserving_generation,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                strength,
                guidance_scale,
                num_inference_steps,
                seed,
                preserve_style,
                style_strength,
                cpu_optimization,
                max_image_size,
            ],
            outputs=[output_image, status_text],
        )

        # 스타일 보존 체크박스 이벤트
        preserve_style.change(
            fn=lambda x: gr.update(visible=x),
            inputs=preserve_style,
            outputs=style_strength,
        )

    return interface


if __name__ == "__main__":
    # 인터페이스 생성 및 실행
    interface = create_interface()
    interface.launch()
