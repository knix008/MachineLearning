import torch
import gradio as gr
from diffusers.utils import load_image, check_min_version
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from PIL import Image, ImageDraw
import datetime
import numpy as np

# 모델 로딩
print("모델 로딩 중...")
controlnet = SD3ControlNetModel.from_pretrained(
    "alimama-creative/SD3-Controlnet-Inpainting", 
    use_safetensors=True, 
    extra_conditioning_channels=1,
    torch_dtype=torch.float16
)

# Inpainting 파이프라인 생성 
pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# 메모리 최적화
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.to("cpu")
print("모델 로딩 완료!")

def resize_image_for_sd3(image, target_size=1024):
    """이미지를 SD3에 적합한 크기로 리사이즈 (16의 배수, 비율 유지)"""
    width, height = image.size
    aspect_ratio = width / height
    
    # 긴 쪽을 기준으로 target_size에 맞춤
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    # 16의 배수로 조정 (비율을 최대한 유지)
    new_width = max(512, (new_width // 16) * 16)
    new_height = max(512, (new_height // 16) * 16)
    
    # 조정된 크기로 비율 재계산하여 더 정확하게 맞춤
    adjusted_ratio = new_width / new_height
    
    # 원본 비율과 차이가 클 경우 더 정확하게 조정
    if abs(aspect_ratio - adjusted_ratio) > 0.1:
        if aspect_ratio > adjusted_ratio:
            # 너비를 늘려야 함
            new_width = min(new_width + 16, target_size + 256)
        else:
            # 높이를 늘려야 함
            new_height = min(new_height + 16, target_size + 256)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def create_center_mask(image, mask_ratio=0.3):
    """이미지 중앙에 원형 마스크 생성"""
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 중앙 좌표
    center_x, center_y = width // 2, height // 2
    
    # 마스크 크기 (이미지 크기의 일정 비율)
    radius = min(width, height) * mask_ratio // 2
    
    # 원형 마스크 그리기 (흰색 = 변경할 영역)
    draw.ellipse([
        center_x - radius, center_y - radius,
        center_x + radius, center_y + radius
    ], fill=255)
    
    return mask

def create_edge_mask(image, edge_threshold=100):
    """이미지의 엣지를 기반으로 마스크 생성"""
    # PIL 이미지를 numpy 배열로 변환
    img_array = np.array(image.convert("RGB"))
    
    # 그레이스케일로 변환
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    # 간단한 엣지 검출 (Sobel-like)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    
    # 패딩을 추가하여 원본 크기 유지
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
    
    # 그래디언트 크기 계산
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 임계값을 초과하는 영역을 마스크로 설정
    mask_array = (gradient_magnitude > edge_threshold).astype(np.uint8) * 255
    
    return Image.fromarray(mask_array, mode="L")

def enhance_image_quality(
    input_image,
    prompt,
    negative_prompt,
    mask_type,
    mask_size,
    num_inference_steps,
    guidance_scale,
    controlnet_conditioning_scale,
    seed,
    target_size,
    progress=gr.Progress()
):
    """고품질 이미지 생성 함수"""
    try:
        progress(0.1, desc="이미지 전처리 중...")
        
        if input_image is None:
            return None, None, "입력 이미지를 업로드해주세요."
        
        # 이미지 크기 조정
        input_image = resize_image_for_sd3(input_image, target_size)
        
        # 마스크 자동 생성
        progress(0.2, desc="마스크 생성 중...")
        if mask_type == "중앙 원형":
            mask_image = create_center_mask(input_image, mask_size)
        elif mask_type == "엣지 기반":
            mask_image = create_edge_mask(input_image, int(mask_size * 255))
        else:  # 전체 영역
            mask_image = Image.new("L", input_image.size, 255)
        
        width, height = input_image.size
        progress(0.3, desc="생성 매개변수 설정 중...")
        
        # 시드 설정
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator("cpu").manual_seed(seed)
        progress(0.5, desc="고품질 이미지 생성 중...")
        
        # 고품질 이미지 생성
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_mask=mask_image,
            control_image=input_image,
            height=height,
            width=width,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generator,
        )
      
        progress(0.9, desc="결과 저장 중...")
        enhanced_image = result.images[0]
        
        # 고품질로 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_inpainting_{timestamp}.png"
        enhanced_image.save(filename, optimize=True, quality=95)
        
        progress(1.0, desc="완료!")
        
        status_message = f"""✅ 이미지 화질 개선 완료!
📁 저장 파일: {filename}
📐 크기: {width}x{height}
🎭 마스크 타입: {mask_type}
🎲 시드: {seed}
⚙️ 추론 스텝: {num_inference_steps}
🎯 가이던스: {guidance_scale}"""
        
        return enhanced_image, mask_image, status_message
        
    except Exception as e:
        return None, None, f"❌ 오류 발생: {str(e)}"

# Gradio 인터페이스 생성
with gr.Blocks(title="SD3 ControlNet Inpainting 화질 개선기", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 SD3 ControlNet Inpainting 화질 개선기
    Stable Diffusion 3 ControlNet을 사용하여 이미지의 특정 부분을 고품질로 인페인팅합니다.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 입력 이미지
            input_image = gr.Image(
                label="📷 입력 이미지",
                type="pil",
                height=500,
                value="default.jpg"
            )
            
            # 프롬프트
            prompt = gr.Textbox(
                label="✨ 프롬프트",
                placeholder="photorealistic, ultra high definition, 8k resolution, masterpiece, best quality",
                lines=3,
                value="photorealistic, ultra high definition, 8k resolution, masterpiece, best quality"
            )
            
            negative_prompt = gr.Textbox(
                label="🚫 네거티브 프롬프트",
                lines=3,
                value="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW, low quality, worst quality, jpeg artifacts"
            )
            
        with gr.Column(scale=1):
            # 마스크 설정
            with gr.Group():
                gr.Markdown("### 🎭 마스크 설정")
                gr.Markdown("""
                **마스크**는 이미지에서 변경할 영역을 지정합니다.
                - **중앙 원형**: 이미지 중앙에 원형 영역을 선택
                - **엣지 기반**: 이미지의 경계선을 감지하여 자동 선택
                - **전체 영역**: 이미지 전체를 변경 대상으로 설정
                """)
                
                mask_type = gr.Radio(
                    choices=["중앙 원형", "엣지 기반", "전체 영역"],
                    value="중앙 원형",
                    label="마스크 타입"
                )
                
                mask_size = gr.Slider(
                    minimum=0.1,
                    maximum=0.8,
                    value=0.3,
                    step=0.05,
                    label="🔍 마스크 크기/강도",
                    info="중앙 원형: 원의 크기 (0.1=작음, 0.8=큼) | 엣지 기반: 감지 민감도 (0.1=세밀, 0.8=큰 경계만)"
                )
            
            # 생성 설정
            with gr.Group():
                gr.Markdown("### ⚙️ 생성 설정")
                gr.Markdown("""
                **생성 품질과 속도를 조절하는 핵심 매개변수들입니다.**
                """)
                
                num_inference_steps = gr.Slider(
                    minimum=20,
                    maximum=100,
                    value=50,
                    step=1,
                    label="🔄 추론 스텝 (높을수록 고품질, 느림)",
                    info="AI가 이미지를 생성하는 반복 횟수. 20=빠름/저품질, 50=균형, 100=느림/고품질"
                )
                
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.1,
                    label="🎯 가이던스 스케일 (프롬프트 충실도)",
                    info="프롬프트를 얼마나 정확히 따를지 결정. 1.0=자유로운 생성, 7.5=균형, 15.0+=프롬프트 엄격 준수"
                )
                
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.95,
                    step=0.05,
                    label="🎮 ControlNet 강도",
                    info="원본 이미지 구조 유지 정도. 0.0=완전 새로운 이미지, 1.0=원본 구조 유지, 2.0=과도한 유지"
                )
                
                target_size = gr.Slider(
                    minimum=512,
                    maximum=1536,
                    value=1024,
                    step=64,
                    label="📐 목표 크기",
                    info="생성될 이미지의 긴 쪽 크기(픽셀). 512=빠름/저해상도, 1024=균형, 1536=느림/고해상도"
                )

                seed = gr.Number(
                    label="🎲 시드 (-1: 랜덤)",
                    value=-1,
                    precision=0,
                    info="동일한 시드로 같은 결과 재현 가능. -1=매번 다른 결과, 고정값=동일 결과"
                )
            
            generate_btn = gr.Button(
                "🚀 고품질 이미지 생성",
                variant="primary",
                size="lg"
            )
    
    with gr.Row():
        with gr.Column():
            output_image = gr.Image(
                label="✨ 개선된 이미지",
                type="pil",
                height=500
            )
            
        with gr.Column():
            generated_mask = gr.Image(
                label="🎭 생성된 마스크",
                type="pil",
                height=500
            )
    
    with gr.Row():
        status_text = gr.Textbox(
            label="📊 생성 정보",
            lines=6,
            max_lines=10
        )
    
    # 이벤트 바인딩
    generate_btn.click(
        fn=enhance_image_quality,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            mask_type,
            mask_size,
            num_inference_steps,
            guidance_scale,
            controlnet_conditioning_scale,
            seed,
            target_size
        ],
        outputs=[output_image, generated_mask, status_text]
    )

# 인터페이스 실행
if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
    )