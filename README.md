# 📖 ⚡ Lit-Story

lit-story is a repository inspired by the remarkable work of Ligtning AI's lit-llama and lit-parrot repositories. This project aims to provide a powerful and interactive console-based and websockets-based tool for generating writing prompts on the fly.

## Features

- Console-based interface for generating writing prompts.
- Websockets ready to be used with your web application
- Customizable temperature and max tokens settings for fine-tuning prompt generation.
- Supports loading StableLM, Pythia, RedPajama, and OpenLLaMA based models.
- Integration with LoRA adapters fine-tuned on the aforementioned base models.
- Adheres to the core principle of openness through clarity, similar to lit-llama and lit-parrot.


## Installation

To install and set up lit-story, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/kstevica/lit-story
cd lit-story
```

lit-story currently relies on FlashAttention from PyTorch nightly. Until PyTorch 2.1 is released you'll need to install nightly manually.
Luckily that is straightforward:

**On CUDA**

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
```

**On CPU (incl Macs)**

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cpu --pre 'torch>=2.1.0dev'
```

All good, now install the dependencies:

### 2. Install the dependecies:

```bash
pip install -r requirements.txt
```

### 3. Base models

Make sure you have converted weights of base models.

- StabilityAI [StableLM](https://github.com/Stability-AI/StableLM)
- EleutherAI [Pythia](https://github.com/EleutherAI/pythia)
- Together [RedPajama-INCITE](https://www.together.xyz/blog/redpajama-models-v1)
- OpenLLaMA 300b [Open-LLaMA](https://huggingface.co/openlm-research/open_llama_7b_preview_300bt)

Convert base weights using either Lit-Parrot (anything except LLaMA based models), either Lit-LLaMA (LLaMA based models).

Lit-Parrot:
```bash
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Base-3B-v1
```

Lit-LLaMA:
```bash
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/hf-llama/7B --output_dir checkpoints/lit-llama/7B --model_size 7B
```



## Run in the console mode


### Anything except LLaMA based models

```bash
python generate_story.py --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Base-3B-v1 \
  --adapter_path path/to/your/adapter 
```


### LLaMA based models

```bash
python generate_story.py --use_llama=True \
  --checkpoint_dir checkpoints//llama/open-7B/lit-llama.pth \
  --tokenizer_path checkpoints/llama/tokenizer.model \
  --adapter_path path/to/your/adapter 
```


## Run in websockets mode

Add run parameters:

```bash
--use_sockets=True --use_port=12345
```


Make sure that you load certificates (in code):

```python
ssl_cert = "ssl/website.pem"
ssl_key = "ssl/website.key"
```

## Use structured prompts

Run with parameter

```bash
--use_alpaca=True
```

and change the code in methods 
```python
generate_output
generate_output_console
```

before the line
```python
encoded =  tokenizer.encode(prompt, device=model.device)
```

to prepare variable *prompt* to be formatted as your fine tuned adapters.


## Acknowledgements

This implementation builds on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [Lit-Parrot](https://github.com/lightning-AI/lit-parrot), and it's powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡.

lit-story builds upon the foundational work of Ligtning AI's lit-llama and lit-parrot repositories. I express my gratitude to the Lightning AI team for their inspiring projects and contributions to the open-source community.

---

Make your writing experience come alive with lit-story! Generate captivating writing prompts, explore different models, and unleash your creativity. Enjoy the power of console-based storytelling with customizable settings and integration capabilities. Let lit-story be your guide on the journey to exceptional writing.
