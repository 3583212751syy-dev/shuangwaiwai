import os
# Use real HuggingFace + proxy + Xet client (hf-mirror routes Xet to an unreachable CDN)
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HUB_ENABLE_HF_XET'] = '1'
os.environ['HF_HOME'] = r'E:\AI\Cache\huggingface'

from huggingface_hub import snapshot_download

print("==> base SD-inpainting (scheduler+unet+model_index) ...")
p1 = snapshot_download(
    repo_id="booksforcharlie/stable-diffusion-inpainting",
    allow_patterns=["model_index.json", "scheduler/**", "unet/**"],
)
print("base ->", p1)

print("==> sd-vae-ft-mse ...")
p2 = snapshot_download(repo_id="stabilityai/sd-vae-ft-mse")
print("vae ->", p2)

print("DONE_WEIGHTS")
