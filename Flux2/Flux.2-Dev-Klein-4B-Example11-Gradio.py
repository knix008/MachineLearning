import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import gradio as gr

class ImageEditor:
    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float32

        print("모델 로딩 중...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B", torch_dtype=self.dtype
        )
        self.pipe = self.pipe.to(self.device)

        # Memory optimization for macOS
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing(1)
        self.pipe.enable_sequential_cpu_offload()
        print("모델 로딩 완료!")

    def edit_image(self, input_image_path, prompt, output_path=None,
                   height=None, width=None, guidance_scale=3.5,
                   num_inference_steps=20, seed=None):
        """
        Edit an image using Klein 4B model

        Args:
            input_image_path: Path to input image
            prompt: Text description of desired changes
            output_path: Path to save edited image (optional)
            height: Output height (None = use input image height)
            width: Output width (None = use input image width)
            guidance_scale: How closely to follow the prompt (1.0-7.0)
            num_inference_steps: Quality vs speed (4-50)
            seed: Random seed for reproducibility
        """
        if not os.path.exists(input_image_path):
            print(f"오류: 입력 이미지를 찾을 수 없습니다: {input_image_path}")
            return None

        # Load input image
        input_image = Image.open(input_image_path).convert("RGB")

        # Use input image dimensions if not specified
        if width is None:
            width = input_image.width
        if height is None:
            height = input_image.height

        print(f"입력 이미지 로드됨: {input_image_path} ({input_image.size})")
        print(f"출력 크기: {width}x{height}")

        # Generate
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        print(f"이미지 편집 중... (steps: {num_inference_steps})")
        image = self.pipe(
            prompt=prompt,
            image=input_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        # Save
        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_name = os.path.splitext(os.path.basename(__file__))[0]
            output_path = f"{base_name}_{timestamp}.png"

        image.save(output_path)
        #print(f"편집된 이미지 저장됨: {output_path}")
        return image

def process_image(editor, image_input, prompt, height, width, guidance_scale, num_inference_steps, seed):
    """
    Process image using the editor
    """
    if image_input is None:
        return None, "오류: 이미지를 입력해주세요."
    
    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."
    
    try:
        # Save temporary input image
        temp_input_path = "temp_input.png"
        image_input.save(temp_input_path)
        
        # Process image with automatic timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{timestamp}.png"
        
        result_image = editor.edit_image(
            input_image_path=temp_input_path,
            prompt=prompt,
            output_path=output_path,
            height=height if height > 0 else None,
            width=width if width > 0 else None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed if seed >= 0 else None
        )
        
        # Clean up
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        
        if result_image:
            return result_image, f"✓ 이미지 편집 완료! 저장됨: {output_path}"
        else:
            return None, "오류: 이미지 편집에 실패했습니다."
    
    except Exception as e:
        return None, f"오류: {str(e)}"

def update_dimensions(image):
    """
    Update height and width sliders based on input image dimensions
    """
    if image is None:
        return 0, 0
    
    if isinstance(image, Image.Image):
        return image.height, image.width
    
    return 0, 0

def main():
    editor = ImageEditor()
    
    # Create Gradio interface
    with gr.Blocks(title="Flux Klein 4B 이미지 편집기") as demo:
        gr.Markdown("# Flux Klein 4B 이미지 편집기")
        gr.Markdown("이미지를 업로드하고 편집하고 싶은 내용을 설명하세요.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="입력 이미지", type="pil", height=800, value="./default.jpg")
                prompt_input = gr.Textbox(
                    label="편집 내용 설명 (영어)",
                    placeholder="예: make the sky blue",
                    value="Highly realistic, 4k, high-quality, high resolution, beautiful full body photo. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing red bikini, walking on a sunny beach. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin appears natural with visible pores.",
                    lines=3
                )
                
                with gr.Accordion("고급 설정", open=False):
                    with gr.Row():
                        height_input = gr.Slider(
                            label="높이 (0=입력 이미지와 동일)",
                            minimum=0,
                            maximum=1536,
                            step=64,
                            value=0
                        )
                        width_input = gr.Slider(
                            label="너비 (0=입력 이미지와 동일)",
                            minimum=0,
                            maximum=1536,
                            step=64,
                            value=0
                        )
                    
                    with gr.Row():
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=7.0,
                            step=0.1,
                            value=3.5
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝",
                            minimum=4,
                            maximum=50,
                            step=1,
                            value=20
                        )
                    
                    seed_input = gr.Slider(
                        label="시드 (-1=랜덤)",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        value=100
                    )
                
                submit_btn = gr.Button("이미지 편집", variant="primary", size="lg")
            
            with gr.Column():
                image_output = gr.Image(label="출력 이미지", height=800)
                status_output = gr.Textbox(label="상태", interactive=False)
        
        # Update dimensions when image changes
        image_input.change(
            fn=update_dimensions,
            inputs=image_input,
            outputs=[height_input, width_input]
        )
        
        # Connect button to processing function
        submit_btn.click(
            fn=process_image,
            inputs=[
                gr.State(editor),
                image_input,
                prompt_input,
                height_input,
                width_input,
                guidance_input,
                steps_input,
                seed_input
            ],
            outputs=[image_output, status_output]
        )
    
    # Launch the interface
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()
