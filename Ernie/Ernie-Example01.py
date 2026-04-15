import torch
from diffusers import ErnieImagePipeline

pipe = ErnieImagePipeline.from_pretrained(
    "Baidu/ERNIE-Image",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()
print("메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CUDA)")

image = pipe(
    prompt="This is a photograph depicting an urban street scene. Shot at eye level, it shows a covered pedestrian or commercial street. Slightly below the center of the frame, a cyclist rides away from the camera toward the background, appearing as a dark silhouette against backlighting with indistinct details. The ground is paved with regular square tiles, bisected by a prominent tactile paving strip running through the scene, whose raised textures are clearly visible under the light. Light streams in diagonally from the right side of the frame, creating a strong backlight effect with a distinct Tyndall effect—visible light beams illuminating dust or vapor in the air and casting long shadows across the street. Several pedestrians appear on the left side and in the distance, some with their backs to the camera and others walking sideways, all rendered as silhouettes or semi-silhouettes. The overall color palette is warm, dominated by golden yellows and dark browns, evoking the atmosphere of dusk or early morning.",
    height=1264,
    width=848,
    num_inference_steps=50,
    guidance_scale=4.0,
    use_pe=True,  # use prompt enhancer
).images[0]

image.save("output.png")
