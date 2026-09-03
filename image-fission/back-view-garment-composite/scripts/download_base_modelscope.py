import os
from modelscope.hub.snapshot_download import snapshot_download

OUT = r'D:\.workbuddy\2026-08-16-00-13-40\image-fission\CatVTON\models\sd_inpainting'
os.makedirs(OUT, exist_ok=True)

allow = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
print("downloading SD-inpainting (fp16) from ModelScope ->", OUT)
path = snapshot_download(
    'AI-ModelScope/stable-diffusion-inpainting',
    cache_dir=OUT,
    allow_patterns=allow,
    revision='master',
)
# CatVTON's from_pretrained expects diffusion_pytorch_model.safetensors/.bin
# ModelScope ships fp16 variants under non-standard names -> rename to standard.
import shutil
for sub in ['unet', 'vae']:
    d = os.path.join(path, sub)
    fp16 = os.path.join(d, 'diffusion_pytorch_model.fp16.safetensors')
    std = os.path.join(d, 'diffusion_pytorch_model.safetensors')
    if os.path.exists(fp16) and not os.path.exists(std):
        shutil.move(fp16, std)
        print("renamed", sub, "fp16 -> standard")
print("DONE_BASE", path)
