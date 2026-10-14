import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

MODEL_PATH = r"C:\MascotAI\models\Wan2.1-T2V-1.3B"
OUTPUT_PATH = r"C:\MascotAI\output\wan_test.mp4"

print("Loading Wan 2.1 T2V 1.3B...")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")

pipe = WanPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
)

# Keep most of the model in system RAM and move
# components to the GPU when needed.
pipe.enable_model_cpu_offload()

# Reduce VAE memory usage.
if hasattr(pipe, "enable_vae_tiling"):
    pipe.enable_vae_tiling()

prompt = (
    "A realistic cinematic shot of a futuristic city at night, "
    "glowing skyscrapers, flying cars moving through the sky, "
    "beautiful reflections on wet streets, smooth camera movement, "
    "photorealistic, detailed lighting"
)

print("Generating video...")
print("This may take a while on an RTX 5060.")

result = pipe(
    prompt=prompt,
    height=480,
    width=832,
    num_frames=49,
    guidance_scale=5.0,
    num_inference_steps=20
)

frames = result.frames[0]

export_to_video(
    frames,
    OUTPUT_PATH,
    fps=16
)

print()
print("VIDEO GENERATION COMPLETE")
print("Saved to:")
print(OUTPUT_PATH)