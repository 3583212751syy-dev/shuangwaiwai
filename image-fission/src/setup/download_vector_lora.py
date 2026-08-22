"""
下载 Doctor Diffusion Vector Art XL LoRA — 治"糊"用
SDXL 写实底子出不了矢量/纹章，加这个 LoRA 直接出"白底黑线"矢量感。
"""
import os
from huggingface_hub import hf_hub_download

PROXY = "http://127.0.0.1:7897"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ[k] = PROXY

LORA_DIR = r"E:\Desktop\双接口\image-fission\ComfyUI\models\loras"
os.makedirs(LORA_DIR, exist_ok=True)

REPO = "DoctorDiffusion/doctor-diffusion-s-controllable-vector-art-xl-lora"
FILE = "DD-vector-v2.safetensors"

if __name__ == "__main__":
    print(f"[DL] {REPO}/{FILE}")
    path = hf_hub_download(
        repo_id=REPO,
        filename=FILE,
        local_dir=LORA_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"[DL] DONE: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")
