"""
下载图裂变 MVP 所需模型到 ComfyUI/models 对应目录。
需在 image-fission/venv 内、且已装 huggingface_hub 后运行：
    python src/setup/download_models.py
走代理：脚本内强制设置 http_proxy/https_proxy。
"""
import os
import zipfile
import sys

for k, v in {"http_proxy": "http://127.0.0.1:7897",
             "https_proxy": "http://127.0.0.1:7897",
             "HTTP_PROXY": "http://127.0.0.1:7897",
             "HTTPS_PROXY": "http://127.0.0.1:7897"}.items():
    os.environ[k] = v

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "ComfyUI", "models")


def d(target_dir, repo, filename, **kw):
    os.makedirs(target_dir, exist_ok=True)
    print(f"[download] {repo}/{filename} -> {target_dir}")
    return hf_hub_download(repo_id=repo, filename=filename,
                           local_dir=target_dir, **kw)


def main():
    # 1) SDXL base checkpoint（CreativeML OpenRAIL-M，允许商用）
    d(os.path.join(MODELS, "checkpoints"),
      "stabilityai/stable-diffusion-xl-base-1.0",
      "sd_xl_base_1.0.safetensors")
    # 2) SDXL VAE
    d(os.path.join(MODELS, "vae"),
      "stabilityai/sdxl-vae", "sdxl_vae.safetensors")
    # 3) IP-Adapter (SDXL) —— 相似度控制
    ip = os.path.join(MODELS, "ipadapter")
    d(ip, "h94/IP-Adapter", "sdxl_models/ip-adapter_sdxl_vit-h.safetensors")
    d(ip, "h94/IP-Adapter", "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors")
    # 图像编码器：IPAdapter_plus 的 unified loader 对 "PLUS" 预设按命名模式
    # (ViT.H.14.*s32B.b79K | ipadapter.*sd15 | sd1.?5.*model) 自动查找 clip_vision。
    # 必须放 clip_vision/ 根目录且文件名匹配 —— 用 sd1.5model.safetensors（dim 1280 的 ViT-H）。
    # h94/IP-Adapter/sdxl_models/image_encoder/ 里的是 ViT-g(dim1664)，与本 plus 模型不匹配，别用。
    enc = os.path.join(MODELS, "clip_vision")
    d(enc, "h94/IP-Adapter", "models/image_encoder/model.safetensors")
    d(enc, "h94/IP-Adapter", "models/image_encoder/config.json")
    # 复制为匹配命名（覆盖任何历史误放的 ViT-g 文件）
    src = os.path.join(enc, "models", "image_encoder", "model.safetensors")
    dst = os.path.join(enc, "sd1.5model.safetensors")
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, dst)
        csrc = os.path.join(enc, "models", "image_encoder", "config.json")
        if os.path.exists(csrc):
            shutil.copyfile(csrc, os.path.join(enc, "sd1.5model.safetensors.config.json"))
        print(f"[copy] 已复制 ViT-H 编码器为 {dst}")
    # 4) InsightFace antelopev2（人脸检测，侵权用）—— 直接下整包 zip 解压最稳
    ins_dir = os.path.join(MODELS, "insightface", "models", "antelopev2")
    os.makedirs(ins_dir, exist_ok=True)
    zip_path = d(ins_dir, "MonsterMMORPG/tools", "antelopev2.zip")
    print(f"[unzip] {zip_path} -> {ins_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ins_dir)
    # zip 内可能套一层 antelopev2/ 目录，扁平化
    nested = os.path.join(ins_dir, "antelopev2")
    if os.path.isdir(nested):
        for f in os.listdir(nested):
            os.rename(os.path.join(nested, f), os.path.join(ins_dir, f))
        os.rmdir(nested)
    print("[check] antelopev2 files:", sorted(os.listdir(ins_dir)))
    print("DONE: 核心模型下载完成。BiRefNet(SDMatte) 与 YOLO-World 权重将在首次运行时由节点/ultralytics 自动下载。")


if __name__ == "__main__":
    main()
