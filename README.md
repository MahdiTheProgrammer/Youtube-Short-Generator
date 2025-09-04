# Youtube-Short-Generator
### In this repo I am trying to build a fully automated pipline for genereating youtube shorts.
## Voice Generation
Tortoise library is used for generating the voice and the codes are available in 
```
tortoise_gen.py
```
## Picture Generation
Each generated short includes a few AI generated pictures. For picture generation I am using:
```
stabilityai/stable-diffusion-2-1
```
model. The code is available in:
```
imgaer.py
```
## Background video
You can choose the background video by moving it in the 
```
vids/
```
directory. The files is in gitignore due to the size of the video but you can receive it by contacting me via mail if needed.

## Text Generation
This pipeline uses RAG and take information from wikipedia. Then the data is fed into 
```
hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
```
Alongside with prompt engineering it generates stories for the pipeline. All the code for text generation is available on: 
```
text-gen-v13.py
```

<br>

## Automation
After downloading the virtual environments you should be able to run this project locally by running:
```
pipe.py
```

## Python Environments
For envs contact me through mail:
```
panahpourmohamadmahdi62@gmail.com
```
## Generation examples
You can find the generated videos in 
```
https://www.youtube.com/@ShortDailyFacts-g1u
```
youtube channel.    