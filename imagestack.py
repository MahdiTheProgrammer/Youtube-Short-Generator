from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip
from PIL import Image
from moviepy.video.fx import resize as resize_fx
import os
import random
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

latest = get_latest_timestamped_dir()

# --- OPTIONAL PATCH for Pillow>=10 (avoid ANTIALIAS error) ---
if not hasattr(Image, 'ANTIALIAS'):
    resize_fx.Image.ANTIALIAS = Image.Resampling.LANCZOS

# --- Paths ---
video_path = str(latest / 'vid_no_pic.mp4')
audio_path = str(latest / 'voice' / 'final_output.wav')
image_folder = str(latest / 'images')
output_path = str(latest / 'Final.mp4')

# --- Load base video and audio ---
video = VideoFileClip(video_path)
audio = AudioFileClip(audio_path)
video = video.set_audio(audio)

# --- Setup ---
duration = video.duration
video_size = video.size

# --- Load and shuffle images ---
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]
random.shuffle(image_files)

# --- Duration per image ---
if not image_files:
    raise ValueError("No PNG images found in the 'images' folder.")
image_duration = duration / len(image_files)

# --- Create image overlay clips ---
image_clips = []
for i, image_path in enumerate(image_files):
    img_clip = (ImageClip(image_path)
                .resize(height=video_size[1] // 2.5)  # scale to half video height
                .set_start(i * image_duration)
                .set_duration(image_duration)
                .set_position(("center", "center"))
                .crossfadein(0.5)
                .crossfadeout(0.5))
    image_clips.append(img_clip)

# --- Combine video and overlays ---
final = CompositeVideoClip([video, *image_clips])
final.write_videofile(output_path, fps=30)
