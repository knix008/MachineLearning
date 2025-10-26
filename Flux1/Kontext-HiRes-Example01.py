import gradio as gr
import torch
from diffusers import FluxKontextPipeline
from PIL import Image

base_model = "black-forest-labs/FLUX.1-Kontext-dev"
lora_model = "chflame163/kontext_hires"

# 프로그램 시작 시 모델을 미리 로딩
model_load_error = None
pipe = None

base_model = "black-forest-labs/FLUX.1-Kontext-dev"
lora_path = "chflame163/kontext_hires"
torch_dtype = torch.bfloat16

# 모델 및 LoRA 미리 로딩
model_load_error = None
pipe = None

try:
	pipe = FluxKontextPipeline.from_pretrained(
		base_model,
		torch_dtype=torch_dtype
	)
	pipe.load_lora_weights(lora_path, prefix=None)
	pipe.enable_model_cpu_offload()
	pipe.enable_sequential_cpu_offload()
	pipe.enable_attention_slicing(1)
	pipe.enable_vae_slicing()
	print("Loading Model is Complete!!!")
except Exception as e:
	model_load_error = str(e)

def resize_keep_ratio(img, max_dim=768):
	w, h = img.size
	if w > max_dim or h > max_dim:
		scale = min(max_dim / w, max_dim / h)
		new_w, new_h = int(w * scale), int(h * scale)
		img = img.resize((new_w, new_h), Image.LANCZOS)
	return img

def upscale_image(input_img, prompt="high quality, detailed", guidance_scale=7.5, num_inference_steps=25):
	if model_load_error:
		return f"모델 로딩 오류: {model_load_error}"
	if pipe is None:
		return "모델이 정상적으로 로딩되지 않았습니다."
	if input_img is None:
		return "이미지를 업로드하세요."
	try:
		img = input_img.convert("RGB")
		img = resize_keep_ratio(img, max_dim=768)
		result = pipe(
			prompt=prompt,
			image=img,
			guidance_scale=guidance_scale,
			num_inference_steps=num_inference_steps,
			generator=torch.Generator(device="cpu").manual_seed(42)
		).images[0]
		return result
	except Exception as e:
		return f"이미지 변환 오류: {e}"

with gr.Blocks() as demo:
	gr.Markdown("# Kontext HiRes 이미지 업스케일러 (FLUX1 + LoRA)")
	with gr.Row():
		with gr.Column():
			try:
				default_img = Image.open("default.jpg")
			except Exception:
				default_img = None
			input_img = gr.Image(type="pil", label="입력 이미지", value=default_img)
			prompt = gr.Textbox(value="high quality, detailed", label="프롬프트")
			guidance_scale = gr.Slider(1.0, 15.0, value=7.5, label="guidance_scale")
			num_inference_steps = gr.Slider(1, 50, value=25, step=1, label="num_inference_steps")
			btn = gr.Button("업스케일 변환")
		with gr.Column():
			output_img = gr.Image(type="pil", label="고해상도 결과")

	btn.click(
		upscale_image,
		inputs=[input_img, prompt, guidance_scale, num_inference_steps],
		outputs=output_img
	)

if __name__ == "__main__":
	demo.launch(inbrowser=True)

