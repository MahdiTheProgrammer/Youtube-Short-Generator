from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import torchaudio
import torch
import re
import sys
import os
from datetime import datetime
from pathlib import Path

# Initialize Tortoise
tts = TextToSpeech()
voice_samples, conditioning_latents = load_voice("daniel")

# Your long input text
text = sys.argv[1]

# Helper: split by sentences while keeping them complete
def split_into_chunks(text, max_chars=100):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ''
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += ' ' + sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

# Split text
chunks = split_into_chunks(text)

# Process each chunk
final_audio = []
for idx, chunk in enumerate(chunks):
    print(f"🔹 Generating chunk {idx+1}/{len(chunks)}...")
    try:
        audio = tts.tts_with_preset(
            text=chunk,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset="fast",
            num_autoregressive_samples=12,
        )
        if audio is not None and audio.shape[-1] > 0:
            final_audio.append(audio)
        else:
            print(f"⚠️ Skipped empty chunk {idx+1}")
    except Exception as e:
        print(f"❌ Failed at chunk {idx+1}: {e}")

# Save final audio
if final_audio:
    combined = torch.cat(final_audio, dim=-1)
    output_dir =Path(datetime.now().strftime("%Y%m%d%H%M")) / "voice"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_output.wav"
    torchaudio.save(str(output_path), combined.squeeze(0).cpu(), 24000)
    print("✅ Saved to final_output.wav")
else:
    print("❌ No audio was successfully generated.")
