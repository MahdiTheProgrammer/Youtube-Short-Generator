import os
import random
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
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

# === CONFIG ===
VIDEO_FOLDER = 'vids'         # Folder where your 16:9 videos are
VOICE_AUDIO = str(latest / 'voice' / 'final_output.wav')          # Your generated Tortoise output
OUTPUT_FILE = str(latest / 'vid_no_pic.mp4')    # Final result

def crop_center_9_16(clip):
    """Crop 16:9 to vertical 9:16 (portrait) format."""
    w, h = clip.size
    new_w = int(h * 9 / 16)
    x_center = w // 2
    x1 = x_center - new_w // 2
    x2 = x_center + new_w // 2
    return clip.crop(x1=x1, x2=x2)

def create_short():
    audio = AudioFileClip(VOICE_AUDIO)
    SHORT_DURATION = audio.duration + 2
    # 1. Pick a random gameplay video
    all_videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.mov', '.mkv'))]
    video_file = os.path.join(VIDEO_FOLDER, random.choice(all_videos))
    clip = VideoFileClip(video_file)

    # 2. Pick a random start time
    max_start = max(0, clip.duration - SHORT_DURATION)
    start_time = random.uniform(0, max_start)
    subclip = clip.subclip(start_time, start_time + SHORT_DURATION)

    # 3. Convert to 9:16 crop
    vertical_clip = crop_center_9_16(subclip)

    # 4. Load voice audio and set to video
    audio = audio.set_duration(min(audio.duration, vertical_clip.duration))
    final = vertical_clip.set_audio(audio)

    # 5. Export
    final.write_videofile(OUTPUT_FILE, codec='libx264', audio_codec='aac', fps=30)

create_short()
