import sys, json
from imager import generate_images

with open(sys.argv[1]) as f:
    prompts = json.load(f)

generate_images(prompts)
