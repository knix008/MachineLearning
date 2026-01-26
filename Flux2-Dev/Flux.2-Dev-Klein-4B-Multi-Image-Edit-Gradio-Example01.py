import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import gradio as gr
import math
import numpy as np

prompt = "Combine these images into a cohesive, artistic composition. Create a seamless blend that incorporates elements from all input images into a single beautiful scene."

class MultiImageEditor:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16

        print("모델 로딩 중...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B", torch_dtype=self.dtype
        )
        self.pipe = self.pipe.to(self.device)
        
        # Memory optimization
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
        self.pipe.enable_sequential_cpu_offload()
        print("모델 로딩 완료!")

    def create_image_grid(self, images, grid_size=None):
        """
        Create a grid/collage from multiple images

        Args:
            images: List of PIL Images
            grid_size: Tuple (cols, rows) or None for auto

        Returns:
            Combined PIL Image
        """
        if not images:
            return None

        n_images = len(images)

        if grid_size is None:
            # Auto-calculate grid size
            cols = math.ceil(math.sqrt(n_images))
            rows = math.ceil(n_images / cols)
        else:
            cols, rows = grid_size

        # Find maximum dimensions
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        # Resize all images to the same size
        resized_images = []
        for img in images:
            resized = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
            resized_images.append(resized)

        # Create the grid
        grid_width = cols * max_width
        grid_height = rows * max_height
        grid_image = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))

        for idx, img in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            x = col * max_width
            y = row * max_height
            grid_image.paste(img, (x, y))

        return grid_image

    def blend_images(self, images, mode="average"):
        """
        Blend multiple images together

        Args:
            images: List of PIL Images
            mode: "average", "overlay", "multiply"

        Returns:
            Blended PIL Image
        """
        if not images:
            return None

        if len(images) == 1:
            return images[0]

        # Find common size (use first image as reference)
        target_width = images[0].width
        target_height = images[0].height

        # Resize all images to same size
        resized_images = [img.resize((target_width, target_height), Image.Resampling.LANCZOS) for img in images]

        if mode == "average":
            # Average blend
            arrays = [np.array(img, dtype=np.float32) for img in resized_images]
            avg_array = np.mean(arrays, axis=0).astype(np.uint8)
            return Image.fromarray(avg_array)

        elif mode == "overlay":
            # Sequential alpha blend
            result = resized_images[0].copy()
            alpha = 1.0 / len(resized_images)
            for img in resized_images[1:]:
                result = Image.blend(result, img, alpha)
            return result

        else:  # multiply
            arrays = [np.array(img, dtype=np.float32) / 255.0 for img in resized_images]
            result = arrays[0]
            for arr in arrays[1:]:
                result = result * arr
            result = (result * 255).astype(np.uint8)
            return Image.fromarray(result)

    def generate_from_multiple(self, images, prompt, output_path=None,
                               combine_mode="grid", height=1024, width=1024,
                               guidance_scale=3.5, num_inference_steps=20, seed=None):
        """
        Generate a new image from multiple input images

        Args:
            images: List of PIL Images
            prompt: Text description for generation
            combine_mode: How to combine images ("grid", "blend_average", "blend_overlay")
            height: Output height
            width: Output width
            guidance_scale: CFG scale
            num_inference_steps: Number of denoising steps
            seed: Random seed

        Returns:
            Generated PIL Image
        """
        if not images:
            print("오류: 입력 이미지가 없습니다.")
            return None

        # Combine input images
        print(f"입력 이미지 {len(images)}개 결합 중... (모드: {combine_mode})")

        if combine_mode == "grid":
            combined_image = self.create_image_grid(images)
        elif combine_mode == "blend_average":
            combined_image = self.blend_images(images, mode="average")
        elif combine_mode == "blend_overlay":
            combined_image = self.blend_images(images, mode="overlay")
        elif combine_mode == "blend_multiply":
            combined_image = self.blend_images(images, mode="multiply")
        else:
            combined_image = self.create_image_grid(images)

        # Resize combined image for the model
        combined_image = combined_image.resize((width, height), Image.Resampling.LANCZOS)
        print(f"결합된 이미지 크기: {combined_image.size}")

        # Generate
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        print(f"새 이미지 생성 중... (steps: {num_inference_steps})")
        result = self.pipe(
            prompt=prompt,
            image=combined_image,
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

        result.save(output_path)
        print(f"생성된 이미지 저장됨: {output_path}")
        return result, combined_image

def process_multiple_images(editor, img1, img2, img3, img4, prompt, combine_mode,
                            height, width, guidance_scale, num_inference_steps, seed):
    """
    Process multiple images using the editor
    """
    # Collect non-None images
    images = []
    for img in [img1, img2, img3, img4]:
        if img is not None:
            images.append(img)

    if len(images) == 0:
        return None, None, "오류: 최소 1개 이상의 이미지를 입력해주세요."

    if not prompt:
        return None, None, "오류: 프롬프트를 입력해주세요."

    try:
        # Generate output filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{timestamp}.png"

        result_image, combined_image = editor.generate_from_multiple(
            images=images,
            prompt=prompt,
            output_path=output_path,
            combine_mode=combine_mode,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed if seed >= 0 else None
        )

        if result_image:
            return combined_image, result_image, f"✓ 이미지 생성 완료! ({len(images)}개 이미지 사용) 저장됨: {output_path}"
        else:
            return None, None, "오류: 이미지 생성에 실패했습니다."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"오류: {str(e)}"

def main():
    editor = MultiImageEditor()

    # Create Gradio interface
    with gr.Blocks(title="Flux Klein 4B 다중 이미지 생성기") as demo:
        gr.Markdown("# Flux Klein 4B Multi-Image Generator")
        gr.Markdown("여러 이미지를 업로드하고, 이를 기반으로 새로운 이미지를 생성합니다.")

        # 입력 이미지들
        gr.Markdown("### 입력 이미지 (1~4개)")
        with gr.Row():
            image_input1 = gr.Image(label="이미지 1", type="pil", height=300)
            image_input2 = gr.Image(label="이미지 2", type="pil", height=300)
            image_input3 = gr.Image(label="이미지 3", type="pil", height=300)
            image_input4 = gr.Image(label="이미지 4", type="pil", height=300)

        # 결합 방식 선택
        with gr.Row():
            combine_mode = gr.Radio(
                label="이미지 결합 방식",
                choices=[
                    ("그리드 (Grid)", "grid"),
                    ("평균 블렌드 (Average Blend)", "blend_average"),
                    ("오버레이 블렌드 (Overlay Blend)", "blend_overlay"),
                    ("곱셈 블렌드 (Multiply Blend)", "blend_multiply")
                ],
                value="grid"
            )

        # 프롬프트 입력
        prompt_input = gr.Textbox(
            label="생성 프롬프트 (영어)",
            placeholder="예: Combine these images into a beautiful landscape, merge these portraits into one person",
            value=prompt,
            lines=3
        )

        # 고급 설정
        with gr.Accordion("고급 설정", open=False):
            with gr.Row():
                height_input = gr.Slider(
                    label="출력 높이",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=1024
                )
                width_input = gr.Slider(
                    label="출력 너비",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=1024
                )

            with gr.Row():
                guidance_input = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=5.0,
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
                maximum=10000,
                step=1,
                value=42
            )

        # 생성 버튼
        submit_btn = gr.Button("이미지 생성", variant="primary", size="lg")

        # 결과 표시
        gr.Markdown("### 결과")
        with gr.Row():
            combined_output = gr.Image(label="결합된 입력 이미지", height=500)
            image_output = gr.Image(label="생성된 이미지", height=500)

        status_output = gr.Textbox(label="상태", interactive=False)

        # Connect button to processing function
        submit_btn.click(
            fn=process_multiple_images,
            inputs=[
                gr.State(editor),
                image_input1,
                image_input2,
                image_input3,
                image_input4,
                prompt_input,
                combine_mode,
                height_input,
                width_input,
                guidance_input,
                steps_input,
                seed_input
            ],
            outputs=[combined_output, image_output, status_output]
        )

        # 사용 예시
        with gr.Accordion("사용 방법", open=False):
            gr.Markdown("""
## 사용 방법

1. **이미지 업로드**: 1~4개의 이미지를 업로드합니다.
2. **결합 방식 선택**:
   - **그리드**: 이미지들을 격자 형태로 배치
   - **평균 블렌드**: 모든 이미지의 픽셀을 평균화
   - **오버레이 블렌드**: 이미지들을 순차적으로 오버레이
   - **곱셈 블렌드**: 픽셀 값을 곱하여 어두운 부분 강조
3. **프롬프트 입력**: 원하는 결과를 영어로 설명합니다.
4. **생성 버튼 클릭**: 이미지 생성을 시작합니다.

## 프롬프트 예시

- `Combine these images into a cohesive artistic composition`
- `Merge these portraits into a single person with features from all`
- `Create a seamless landscape by blending these scenes`
- `Transform these images into a surreal dreamscape`
- `Fuse these objects into one unified design`
            """)

    # Launch the interface
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()
