import os
from modelscope.hub.snapshot_download import snapshot_download

local = "E:/Desktop/双接口/image-fission/models/sd15_inpaint"
os.makedirs(local, exist_ok=True)
# 下载完整 SD1.5-inpaint 权重（约 8G，含 fp16 冗余变体；
# 注：modelscope 1.40 的 ignore_file_pattern 对正在进行的下载未必生效，故默认全量）。
# 首次运行需联网；权重不入库（见 image-fission/.gitignore 的 models/）。
print("downloading AI-ModelScope/stable-diffusion-inpainting (full weights) ->", local, flush=True)
path = snapshot_download(
    "AI-ModelScope/stable-diffusion-inpainting",
    local_dir=local,
    ignore_file_pattern=[r"\.fp16\."],
)
print("DONE", path, flush=True)
