import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os

class ImageEditor:
    def __init__(self):
        self.device = "mps"
        self.dtype = torch.bfloat16

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
            output_path = f"edited_{timestamp}.png"

        image.save(output_path)
        print(f"편집된 이미지 저장됨: {output_path}")
        return image

def main():
    editor = ImageEditor()

    print("\n=== Flux Klein 4B 이미지 편집기 ===\n")

    # Interactive mode
    while True:
        print("\n옵션:")
        print("1. 이미지 편집")
        print("2. 종료")
        choice = input("\n선택하세요 (1-2): ").strip()

        if choice == "2":
            print("종료합니다.")
            break

        if choice == "1":
            input_path = input("입력 이미지 경로: ").strip()
            if not input_path:
                print("경로를 입력해주세요.")
                continue

            prompt = input("편집 내용 설명 (영어): ").strip()
            if not prompt:
                print("프롬프트를 입력해주세요.")
                continue

            # Optional parameters
            use_advanced = input("고급 설정 사용? (y/n, 기본값: n): ").strip().lower()

            if use_advanced == 'y':
                try:
                    height_input = input("높이 (엔터=입력 이미지와 동일): ").strip()
                    width_input = input("너비 (엔터=입력 이미지와 동일): ").strip()
                    height = int(height_input) if height_input else None
                    width = int(width_input) if width_input else None
                    guidance = float(input("Guidance scale (1.0-7.0, 기본값: 3.5): ").strip() or "3.5")
                    steps = int(input("추론 스텝 (4-50, 기본값: 20): ").strip() or "20")
                    seed_input = input("시드 (엔터=랜덤): ").strip()
                    seed = int(seed_input) if seed_input else None
                except ValueError:
                    print("잘못된 입력입니다. 기본값을 사용합니다.")
                    height, width, guidance, steps, seed = None, None, 3.5, 20, None
            else:
                height, width, guidance, steps, seed = None, None, 3.5, 20, None

            output = input("출력 파일명 (엔터=자동 생성): ").strip() or None

            editor.edit_image(
                input_image_path=input_path,
                prompt=prompt,
                output_path=output,
                height=height,
                width=width,
                guidance_scale=guidance,
                num_inference_steps=steps,
                seed=seed
            )

if __name__ == "__main__":
    main()
