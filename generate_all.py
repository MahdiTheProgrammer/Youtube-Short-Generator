# generate_all.py
import importlib.util
import os
import subprocess
import json


# Load variables from story_inputs.py
spec = importlib.util.spec_from_file_location("story_inputs", "story_inputs.py")
story_inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(story_inputs)

# How many stories to process (automatically detects)
story_ids = [k.split('_')[1] for k in dir(story_inputs) if k.startswith("story_")]
story_ids = sorted(set(story_ids))

# Loop through each story
for sid in story_ids:
    story_text = getattr(story_inputs, f"story_{sid}")
    pic_prompts = getattr(story_inputs, f"pic_prompts_{sid}")

    print(f"🔄 Generating story {sid}...")

    # Save inputs to intermediate files
#    with open("current_story.txt", "w") as f:
#        f.write(story_text)
    with open("tmp_prompts.json", "w") as f:
        json.dump(pic_prompts,f)

    # Activate env & run each component
#    os.system(f'source ~/tortoise-env/bin/activate && python tortoise_gen.py "{story_text}"')
    subprocess.run(
    	f'bash -c "source ~/tortoise-env/bin/activate && python tortoise_gen.py \\"{story_text}\\""',
    	shell=True
    )
    os.system("python shorter.py")
#    os.system("source ~/stable-env/bin/activate && python imager.py")
    subprocess.run(["/home/mahdi/imager310/bin/python", "run_imager.py", "tmp_prompts.json"])
    subprocess.run(["/home/mahdi/tortoise-env/bin/python", "imagestack.py"])
#    os.system("source ~/tortoise-env/bin/activate && python imagestack.py")

    print(f"✅ Done story {sid}!\n")
