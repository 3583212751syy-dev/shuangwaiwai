"""
预下载 BiRefNet-matting 模型到 ComfyUI/models/RMBG/BiRefNet/。
ComfyUI-RMBG 节点（BiRefNetRMBG）需要：birefnet.py + BiRefNet_config.py
+ BiRefNet-matting.safetensors + config.json（从 HF 仓库 1038lab/BiRefNet 下载）。
节点首次运行会自动下载，但 ComfyUI 进程无代理会失败，故这里手动预下载。
走本机 GW 代理；不要用 hf-mirror（308 跳回 huggingface.co 会触发元数据校验错误）。
"""
import os

PROXY = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY

from huggingface_hub import hf_hub_download

DST = r"E:\Desktop\双接口\image-fission\ComfyUI\models\RMBG\BiRefNet"
os.makedirs(DST, exist_ok=True)

REPO = "1038lab/BiRefNet"
FILES = [
    "birefnet.py",
    "BiRefNet_config.py",
    "BiRefNet-matting.safetensors",
    "config.json",
]


def main():
    for f in FILES:
        print(f"[DL] {f} ...")
        p = hf_hub_download(repo_id=REPO, filename=f, local_dir=DST)
        print(f"[OK] {os.path.getsize(p)/1024/1024:.1f} MB -> {p}")
    print("[DONE] BiRefNet-matting 模型就绪")


if __name__ == "__main__":
    main()
