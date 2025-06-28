import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
from huggingface_hub import snapshot_download

# 📁 Save path
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ✅ Prompt
prompt = (
    "indo-realism, a graceful Indian woman named Divya from Delhi, early 20s, "
    "radiant medium brown skin, expressive almond-shaped brown eyes, straight long jet-black hair, "
    "symmetrical oval face, slim figure, soft pink lips, minimal natural makeup, "
    "wearing a pastel yellow salwar suit with a light dupatta, delicate gold earrings, "
    "DSLR portrait style, cinematic natural lighting, outdoor Indian cafe background, "
    "photorealistic 4K, intelligent and confident, gentle smile, elegant posture, "
    "Instagram influencer vibe, soft bokeh, depth of field"
)

negative_prompt = (
    "cartoon, anime, 3d render, blurry, painting, overexposed skin, exaggerated expression, "
    "too much makeup, distorted face, deformed hands, low-res, bold pose, seductive outfit, glitch, error"
)

# 🎛️ Generation config
seed = 42
guidance_scale = 8.5
num_steps = 50
device = "mps"  # Use Metal backend on Mac M2

# 🧠 Load base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    # "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# # Load LoRA using custom method since diffusers 0.25+ and peft need SDXL-compatible models
# # Download LoRA manually
# lora_path = snapshot_download("prithivMLmods/Flux.1-Dev-Indo-Realism-LoRA")
# pipe.load_lora_weights(lora_path)

pipe.to(device)

# 🔁 Seeded generation
generator = torch.Generator(device=device).manual_seed(seed)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_steps,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]

# 💾 Save
filename = f"divya_seed{seed}.jpg"
filepath = os.path.join(output_dir, filename)
image.save(filepath)
print(f"✅ Image saved at: {filepath}")