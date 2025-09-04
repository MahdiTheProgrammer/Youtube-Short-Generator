# generate_all.py
"""
Pipeline runner:
1) Calls your local RAG story generator script (creates narration + image prompts in ./outputs)
2) Picks up the newest generated pair
3) Runs your TTS + image + stacking steps automatically

Edit the CONFIG paths below to match your environment.
Run:
    python generate_all.py                 # generate 1 story and make a video
    python generate_all.py 3               # generate 3 stories in a row and make 3 videos
"""

import sys
import json
import time
from pathlib import Path
import subprocess
from typing import List, Tuple
import ast  # <-- Add this import

# =========================
# CONFIG — change as needed
# =========================
# Which generator script to call (must save files into OUTPUT_DIR)
GENERATOR_SCRIPT = Path("text-gen-v13.py")  # or Path("yt_short_rag_local_llama.py")
OUTPUT_DIR = Path("outputs")

# Your environments / executables
# If you prefer to rely on shell activation, keep TORTOISE_ACTIVATE non-empty.
TORTOISE_ACTIVATE = "source ~/tortoise-env/bin/activate"
TORTOISE_PY = "/home/mahdi/tortoise-env/bin/python"  # used for scripts we want to run with that env
IMAGER_PY = "/home/mahdi/imager310/bin/python"

# Your component scripts
TORTOISE_GEN = "tortoise_gen.py"  # expects: python tortoise_gen.py "<story>"
SHORTER = "shorter.py"            # trims/edits audio if needed
IMAGER = "run_imager.py"          # expects: python run_imager.py tmp_prompts.json
IMAGESTACK = "imagestack.py"      # stitches frames to video, uses tortoise env python

TMP_PROMPTS_JSON = Path("tmp_prompts.json")

# =========================
# Helpers
# =========================

def run_bash(cmd: str) -> int:
    """Run a bash command with login shell features (needed for 'source')."""
    return subprocess.run(["bash", "-lc", cmd], check=False).returncode


def run_python(py_exe: str, args: List[str]) -> int:
    return subprocess.run([py_exe, *args], check=False).returncode


def list_pairs(out_dir: Path) -> List[Tuple[Path, Path]]:
    """Return list of (narration_path, images_path) pairs sorted by mtime desc."""
    narr = list(out_dir.glob("*_narration.txt"))
    imgs = {p.name.replace("_images.txt", ""): p for p in out_dir.glob("*_images.txt")}
    pairs = []
    for n in narr:
        base = n.name.replace("_narration.txt", "")
        if base in imgs:
            pairs.append((n, imgs[base]))
    pairs.sort(key=lambda t: max(t[0].stat().st_mtime, t[1].stat().st_mtime), reverse=True)
    return pairs


def read_narration(narration_path: Path) -> str:
    return narration_path.read_text(encoding="utf-8").strip()


def read_image_prompts(images_path: Path) -> List[str]:
    """Parse the Python list from the images prompt file."""
    text = images_path.read_text(encoding="utf-8")
    # Find the first '=' and parse the right-hand side as a Python list
    try:
        rhs = text.split('=', 1)[1].strip()
        prompts = ast.literal_eval(rhs)
        if isinstance(prompts, list):
            return [str(p).strip() for p in prompts if str(p).strip()]
    except Exception as e:
        print(f"Error parsing image prompts from {images_path}: {e}")
    return []


def generate_one_story() -> Tuple[Path, Path]:
    """Run the generator script once and return (narration_path, images_path) for the newest pair."""
    before = {p.name for p in OUTPUT_DIR.glob("*.txt")}
    # Call the generator with your system python; the generator itself will use its own model env
    code = run_python(sys.executable, [str(GENERATOR_SCRIPT)])
    if code != 0:
        raise RuntimeError("Story generator failed")
    # Wait a moment for filesystem flush
    time.sleep(0.5)
    # Find the newest pair that wasn't present before
    new_pairs = [pair for pair in list_pairs(OUTPUT_DIR)
                 if pair[0].name not in before and pair[1].name not in before]
    if new_pairs:
        return new_pairs[0]
    # Fallback: just take the newest pair overall
    pairs = list_pairs(OUTPUT_DIR)
    if not pairs:
        raise FileNotFoundError("No narration/images pair found in outputs/")
    return pairs[0]


def make_video_from_pair(narr: Path, imgs: Path) -> None:
    story_text = read_narration(narr)
    pic_prompts = read_image_prompts(imgs)

    # Save prompts to JSON for your imager step
    TMP_PROMPTS_JSON.write_text(json.dumps(pic_prompts, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"🔊 TTS for: {narr.name}")
    if TORTOISE_ACTIVATE:
        safe_story = story_text.replace("\\", "\\\\").replace('"', '\\"')
        run_bash(f'{TORTOISE_ACTIVATE} && python {TORTOISE_GEN!s} "{safe_story}"')
    else:
        run_python(TORTOISE_PY, [TORTOISE_GEN, story_text])

    print("✂️  Post-process audio")
    # shorter.py usually runs in same env as tortoise; use explicit tortoise python to be safe
    run_python(TORTOISE_PY, [SHORTER])

    print("🖼️  Generating images")
    run_python(IMAGER_PY, [IMAGER, str(TMP_PROMPTS_JSON)])

    print("🎞️  Stacking frames to video")
    run_python(TORTOISE_PY, [IMAGESTACK])


# =========================
# Main
# =========================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # How many stories to generate this run
    n = 1
    if len(sys.argv) > 1:
        try:
            n = max(1, int(sys.argv[1]))
        except ValueError:
            pass

    for i in range(n):
        print(f"\n============================\n🚀 Pipeline run {i+1}/{n}\n============================")
        narr_path, img_path = generate_one_story()
        print(f"Found outputs:\n  - {narr_path}\n  - {img_path}")
        make_video_from_pair(narr_path, img_path)
        print("✅ Done!\n")
