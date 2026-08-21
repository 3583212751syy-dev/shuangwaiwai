import os, sys, shutil
from huggingface_hub import hf_hub_download
CV = r"E:\Desktop\双接口\image-fission\ComfyUI\models\clip_vision"
tgtdir = os.path.join(CV, "_vit_h_complete")
os.makedirs(tgtdir, exist_ok=True)
print("[DL] 开始下载 ViT-H 编码器(完整, 含 xet 重组与校验)...")
path = hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/image_encoder/model.safetensors",
    local_dir=tgtdir,
    local_dir_use_symlinks=False,
)
print("[DL] 下载完成:", path, "size=", os.path.getsize(path))
dest = os.path.join(CV, "sd1.5model.safetensors")
shutil.copyfile(path, dest)
print("[DL] 已复制为 sd1.5model.safetensors size=", os.path.getsize(dest))
