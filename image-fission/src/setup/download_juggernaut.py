"""
下载 Juggernaut XL v9（写实基底，替代 SDXL 1.0 通用底模）到 ComfyUI checkpoints。
走本机 GW 代理（127.0.0.1:7897，香港节点）。
协议：CreativeML OpenRAIL-M（可商用），与本项目商用红线一致。
"""
import os
import sys

# ---- 代理（本机 GW，香港节点）----
PROXY = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
# 注意：不要用 hf-mirror.com —— 它会 308 跳回 huggingface.co，导致 huggingface_hub
# 元数据校验报 "Distant resource does not seem to be on huggingface.co"。
# 直连 huggingface.co 走代理即可，xet 分块由 huggingface_hub 正确处理。

from huggingface_hub import hf_hub_download

CKPT_DIR = r"E:\Desktop\双接口\image-fission\ComfyUI\models\checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

REPO = "RunDiffusion/Juggernaut-XL-v9"
FILE = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"


def main():
    print(f"[DL] 从 {REPO} 下载 {FILE} ...")
    print(f"[DL] 目标目录: {CKPT_DIR}")
    path = hf_hub_download(
        repo_id=REPO,
        filename=FILE,
        local_dir=CKPT_DIR,
    )
    size = os.path.getsize(path)
    print(f"[DL] 完成: {path} ({size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
