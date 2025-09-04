from diffusers import StableDiffusionPipeline
import torch
import os
from pathlib import Path

def get_latest_timestamped_dir(base_path='.'):
    base = Path(base_path)
    # Only select directories with purely numeric names of the correct length (e.g., 12 for 'YYYYMMDDHHMM')
    timestamped_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 12]
    if not timestamped_dirs:
        return None  # or raise an error if you want

    # Sort them by name (chronological order because of format)
    latest_dir = max(timestamped_dirs, key=lambda d: d.name)
    return latest_dir

# Usage
latest = get_latest_timestamped_dir()




# Load pipeline once (globally)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

def generate_images(prompts, prefix="generated"):
    """
    Generates images from a list of prompts using Stable Diffusion.

    Args:
        prompts (List[str]): List of prompt strings.
        output_folder (str): Where to save generated images.
        prefix (str): Filename prefix for the images.
    """
    latest = get_latest_timestamped_dir()
    output_folder = str(latest / 'images')
    os.makedirs(output_folder, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"🔹 Generating image {i+1}/{len(prompts)}: {prompt}")
        image = pipe(prompt, height=768, width=768).images[0]  # SD 2.1 supports 768x768
        image.save(os.path.join(output_folder, f"{prefix}_{i+1}.png"))

