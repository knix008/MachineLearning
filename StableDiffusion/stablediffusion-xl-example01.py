import torch
from diffusers import DiffusionPipeline

def generate_high_resolution_image(prompt, output_path="output.png", resolution=(1024, 1024)):
    # Stable Diffusion XL(혹은 최신 고해상도 지원 모델) 사용 권장
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"  # HuggingFace Model Hub에서 가져옴

    base = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        model_id,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")
    
        # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8

    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    # 이미지 저장
    image.save(output_path)
    print(f"이미지가 저장되었습니다: {output_path}")
    
def main():
    prompt = "a futuristic cityscape at night, ultra high resolution, photorealistic, 8k"
    generate_high_resolution_image(prompt, output_path="high_resolution_result.png", resolution=(1024, 1024))

if __name__ == "__main__":
    print("Stable Diffusion XL 예제 시작")
    main()
    