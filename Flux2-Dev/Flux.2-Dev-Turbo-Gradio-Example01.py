import torch
import gradio as gr
from diffusers import Flux2Pipeline
from datetime import datetime
import signal
import sys
import gc

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

device = "cuda"
device_type = torch.bfloat16

print("모델 로딩 중...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=device_type
)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()

pipe.load_lora_weights(
    "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
)
print("모델 로딩 완료!")

prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style korean girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a half-body shot with perfect anatomy. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

#prompt = "Highly realistic, 4k, high-quality, high resolution, beautiful full body korean woman model photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin appears natural with visible pores. Orange hue, solid orange backdrop, using a camera setup that mimics a large aperture, f/1.4 --ar 9:16 --style raw."


def clear_cuda_memory():
    """CUDA 메모리 정리 (파이프라인 유지)"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("CUDA 메모리 정리 완료")


def generate_image(prompt, guidance_scale, height, width, num_steps, seed):
    if seed == -1:
        actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        actual_seed = int(seed)

    generator = torch.Generator(device="cpu").manual_seed(actual_seed)

    try:
        image = pipe(
            prompt=prompt,
            sigmas=TURBO_SIGMAS[:num_steps] if num_steps <= len(TURBO_SIGMAS) else None,
            guidance_scale=guidance_scale,
            height=int(height),
            width=int(width),
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]

        # Save image with script name, steps, user input seed, and guidance scale
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Flux.2-Dev-Turbo-Gradio-Example01_{timestamp}_steps{int(num_steps)}_seed{int(seed)}_guidance{guidance_scale}.jpg"
        image.save(output_filename)
        print("이미지가 저장되었습니다:", output_filename)

        return image, f"저장됨: {output_filename}"

    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM 오류 발생! VRAM 정리 중...")
        clear_cuda_memory()
        return None, "⚠️ CUDA OOM 오류: GPU 메모리 부족. 해상도를 낮추거나 스텝 수를 줄여주세요. (VRAM 정리 완료)"

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("RuntimeError OOM 오류 발생! VRAM 정리 중...")
            clear_cuda_memory()
            return None, "⚠️ OOM 오류: GPU 메모리 부족. 해상도를 낮추거나 스텝 수를 줄여주세요. (VRAM 정리 완료)"
        else:
            clear_cuda_memory()
            return None, f"⚠️ 오류 발생: {str(e)}"


with gr.Blocks(title="FLUX.2-dev Turbo") as demo:
    gr.Markdown("# FLUX.2-dev Turbo Image Generator")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=4,
                value=prompt
            )

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=10.0, value=2.5, step=0.1,
                    label="Guidance Scale"
                )
                num_steps = gr.Slider(
                    minimum=4, maximum=20, value=8, step=1,
                    label="Inference Steps"
                )

            with gr.Row():
                width = gr.Slider(
                    minimum=512, maximum=1536, value=768, step=64,
                    label="Width"
                )
                height = gr.Slider(
                    minimum=512, maximum=1536, value=1024, step=64,
                    label="Height"
                )

            seed = gr.Slider(
                minimum=-1, maximum=1000, value=42, step=1,
                label="Seed (-1 for random)"
            )

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil", height=800)
            status = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, guidance_scale, height, width, num_steps, seed],
        outputs=[output_image, status]
    )

def cleanup():
    """리소스 정리 함수"""
    print("\n리소스 정리 중...")

    # Gradio 서버 종료
    try:
        demo.close()
        print("Gradio 서버 종료 완료")
    except Exception as e:
        print(f"Gradio 종료 중 오류: {e}")

    # 파이프라인 메모리 해제
    global pipe
    try:
        del pipe
        print("파이프라인 메모리 해제 완료")
    except Exception as e:
        print(f"파이프라인 해제 중 오류: {e}")

    # CUDA 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA 캐시 정리 완료")

    # 가비지 컬렉션 실행
    gc.collect()
    print("가비지 컬렉션 완료")
    print("모든 리소스 정리 완료!")


def signal_handler(sig, frame):
    """시그널 핸들러 (Ctrl+C)"""
    print("\n\nKeyboard Interrupt 감지!")
    cleanup()
    sys.exit(0)


if __name__ == "__main__":
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)

    try:
        demo.launch(inbrowser=True)
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"오류 발생: {e}")
        cleanup()
    finally:
        print("프로그램 종료")
