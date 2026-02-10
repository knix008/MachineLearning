import torch
import gradio as gr
from diffusers import Flux2Pipeline
from datetime import datetime
import os
import gc
import signal
import sys

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

DEFAULT_PROMPT = "The image is a high-quality,photorealistic cosplay portrait of a young Asian woman with a soft, idol aesthetic.Physical Appearance: Face: She has a fair,clear complexion.She is wearing striking bright blue contact lenses that contrast with her dark hair.Her expression is innocent and curious,looking directly at the camera with her index finger lightly touching her chin.Hair: She has long,straight jet-black hair with thick,straight-cut bangs (fringe) that frame her face.Attire (Blue & White Bunny Theme): Headwear: She wears tall,upright blue fabric bunny ears with white lace inner lining and a delicate white lace headband base,accented with a small white bow.Outfit: She wears a unique blue denim-textured bodysuit.It features a front zipper,silver buttons,and thin silver chains draped across the chest.The sides are constructed from semi-sheer white lace.Accessories: Around her neck is a blue bow tie attached to a white collar.She wears long,white floral lace fingerless sleeves that extend past her elbows,finished with blue cuffs and small black decorative ribbons.Legwear: She wears white fishnet stockings held up by blue and white ruffled lace garters adorned with small white bows.Pose: She is sitting gracefully on the edge of a light-colored,vintage-style bed or cushioned bench.Her body is slightly angled toward the camera,creating a soft and inviting posture.Setting & Background: Location: A bright,high-key studio set designed to look like a clean,airy bedroom.Background: The background is dominated by large windows with white vertical blinds or curtains,allowing soft,diffused natural-looking light to flood the scene.The background is softly blurred (bokeh).Lighting: The lighting is bright,soft,and even,minimizing harsh shadows and giving the skin a glowing,porcelain appearance.Flux Prompt Prompt: A photorealistic,high-quality cosplay portrait of a beautiful Asian woman dressed in a blue and white bunny girl outfit.She has long straight black hair with hime-cut bangs and vibrant blue eyes.She wears tall blue bunny ears with white lace trim,a blue denim-textured bodysuit with a front zipper and white lace side panels,a blue bow tie,and long white lace sleeves.She is sitting on a white bed in a bright,sun-drenched room with soft-focus white curtains.She poses with a finger to her chin,looking at the camera with a soft,innocent expression.8k resolution,high-key lighting,cinematic soft focus,detailed textures of denim and lace,gravure photography style.Key Stylistic Keywords Blue bunny girl,denim cosplay,white lace,high-key lighting,blue contact lenses,black hair with bangs,fishnet stockings,airy atmosphere,photorealistic,innocent and alluring,studio photography."

# Global variables for model
DEVICE = "mps"
DTYPE = torch.bfloat16
pipe = None


def load_model():
    """Load and initialize the Flux2 pipeline with Turbo LoRA weights."""
    global pipe

    print("모델 로딩 중...")
    pipe = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", torch_dtype=DTYPE
    )

    pipe.load_lora_weights(
        "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
    )
    print("모델 로딩 완료!")
    return pipe


def generate_image(prompt, guidance_scale, height, width, num_steps, seed):
    """Generate image from text prompt and return for UI display."""
    global pipe

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        # Setup seed
        if seed == -1:
            actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            actual_seed = int(seed)

        generator = torch.Generator(device="cpu").manual_seed(actual_seed)

        # Use turbo sigmas when steps fit within the precomputed table
        sigmas = (
            TURBO_SIGMAS[: int(num_steps)]
            if int(num_steps) <= len(TURBO_SIGMAS)
            else None
        )

        print(
            f"이미지 생성 중... (steps: {int(num_steps)}, size: {int(width)}x{int(height)}, seed: {actual_seed})"
        )

        image = pipe(
            prompt=prompt,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_steps),
            generator=generator,
        ).images[0]

        # Save with timestamp and parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_filename = f"{base_name}_{timestamp}_steps{int(num_steps)}_seed{int(seed)}_guidance{guidance_scale}.png"
        image.save(output_filename)
        print(f"이미지가 저장되었습니다: {output_filename}")

        gc.collect()

        return image, f"✓ 저장됨: {output_filename}"

    except Exception as e:
        gc.collect()
        return None, f"오류: {str(e)}"


def main():
    load_model()

    with gr.Blocks(title="FLUX.2-dev Turbo (MacOS)") as demo:
        gr.Markdown("# FLUX.2-dev Turbo Image Generator (MacOS)")
        gr.Markdown("텍스트 설명을 입력하여 이미지를 생성하세요.")

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    value=DEFAULT_PROMPT,
                    lines=5,
                )

                with gr.Accordion("고급 설정", open=True):
                    with gr.Row():
                        guidance_input = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=2.5,
                            step=0.1,
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                        )
                        steps_input = gr.Slider(
                            minimum=4,
                            maximum=20,
                            value=8,
                            step=1,
                            label="Inference Steps",
                            info="생성 품질 (높을수록 고품질, 느림)",
                        )

                    with gr.Row():
                        height_input = gr.Slider(
                            minimum=512,
                            maximum=2048,
                            value=1536,
                            step=64,
                            label="Height",
                        )
                        width_input = gr.Slider(
                            minimum=512,
                            maximum=2048,
                            value=768,
                            step=64,
                            label="Width",
                        )

                    seed_input = gr.Slider(
                        minimum=-1,
                        maximum=1000,
                        value=42,
                        step=1,
                        label="Seed (-1 for random)",
                    )

                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil", height=800)
                status_output = gr.Textbox(label="Status", interactive=False)

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                guidance_input,
                height_input,
                width_input,
                steps_input,
                seed_input,
            ],
            outputs=[output_image, status_output],
        )

    def cleanup():
        """리소스 정리 함수"""
        print("\n리소스 정리 중...")

        try:
            demo.close()
            print("Gradio 서버 종료 완료")
        except Exception as e:
            print(f"Gradio 종료 중 오류: {e}")

        global pipe
        try:
            del pipe
            print("파이프라인 메모리 해제 완료")
        except Exception as e:
            print(f"파이프라인 해제 중 오류: {e}")

        gc.collect()
        print("가비지 컬렉션 완료")
        print("모든 리소스 정리 완료!")

    def signal_handler(sig, frame):
        """시그널 핸들러 (Ctrl+C)"""
        print("\n\nKeyboard Interrupt 감지!")
        cleanup()
        sys.exit(0)

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


if __name__ == "__main__":
    main()
