"""v116 裂变加强：基于 v115 已验证管线（img2img 锁构图 + IPAdapter 锁配色），
denoise 提高到 0.70 / 0.74 双档 → 内容裂变更强更丰富；prompt 增加装饰美感词。

用户反馈（2026-08-26）：「效果还行，裂变效果再多一点好看一点」
- 保持：颜色不变（IPAdapter 0.55 锁黑底白线）/ 构图不变（img2img 原图 1024² latent 起步）
- 增强：denoise 0.58/0.66 → 0.70/0.74（更多重画空间）
- 增强：prompt 加 intricate/ornate/beautiful 等美感描述

用法：python src/smoke_v116_illust.py
"""
import time, requests, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REF_IMG = "pinterest_illust_1_hd.png"

POS_TEMPLATE = (
    "t-shirt graphic design, elegant peacock silhouette with elaborate flowing tail feathers, "
    "same composition and layout as reference image, pure black background, white linework only, "
    "high contrast, "
    "REDESIGNED internal details, new intricate ornamental patterns, "
    "beautifully reimagined feather textures, elegant decorative swirls, "
    "rich ornate flourishes, refined artistic variations, "
    "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
    "fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain"
)

NEGATIVE = (
    "no gray, no midtones, no color, no gradient, no soft shading, no halftone, "
    "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, "
    "margin, border, frame, padding, empty corners, letterbox, white border, smudge, "
    "text, letters, watermark, duplicate image, exact copy"
)


def build_workflow(seed, denoise):
    """同 v115 结构（已验证），仅 denoise 参数化 + SaveImage 节点 14"""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMG}}
    g["3"] = {"class_type": "ImageScale", "inputs": {
        "image": ["2", 0],
        "width": 1024,
        "height": 1024,
        "upscale_method": "lanczos",
        "crop": "center",
    }}
    g["4"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader",
              "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0],
        "ipadapter": ["5", 1],
        "image": ["3", 0],
        "weight": 0.55,
        "weight_type": "style transfer",
        "combine_embeds": "average",
        "start_at": 0.0,
        "end_at": 0.85,
        "noise": 0.05,
        "embeds_scaling": "V only",
    }}
    g["7"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": POS_TEMPLATE}}
    g["8"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": NEGATIVE}}
    g["9"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0],
        "positive": ["7", 0],
        "negative": ["8", 0],
        "latent_image": ["4", 0],
        "seed": seed,
        "steps": 45,
        "cfg": 7.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": denoise,
    }}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0],
        "positive": ["7", 0],
        "negative": ["8", 0],
        "latent_image": ["9", 0],
        "seed": seed + 1,
        "steps": 40,
        "cfg": 7.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 0.20,
    }}
    g["11"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["13"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["12", 0], "image": ["11", 0]}}
    g["14"] = {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0],
                          "filename_prefix": f"v116_d{str(denoise).replace('.', '')}"}}
    return g


def run_one(seed, denoise, out_base, tag):
    workflow = build_workflow(seed, denoise)
    payload = {"prompt": workflow, "client_id": f"v116_{int(time.time())}"}
    r = requests.post(f"{COMFYUI}/prompt", json=payload, timeout=10)
    if r.status_code != 200:
        print(f"[ERR] submit {tag} failed: {r.status_code} {r.text[:300]}", flush=True)
        return False
    pid = r.json().get("prompt_id")
    print(f"  [{tag}] prompt_id={pid}", flush=True)

    for i in range(120):
        time.sleep(5)
        try:
            q = requests.get(f"{COMFYUI}/history/{pid}", timeout=5).json()
            if pid in q:
                rec = q[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("14", {}).get("images", [])
                    if imgs:
                        fname = imgs[0]["filename"]
                        sub = imgs[0].get("subfolder", "")
                        url = f"{COMFYUI}/view?filename={fname}&type=output&subfolder={sub}"
                        data = requests.get(url, timeout=30).content
                        out = out_base / f"{tag}.jpg"
                        out.write_bytes(data)
                        print(f"  [{tag}] OK {out} ({out.stat().st_size/1024/1024:.1f}MB)", flush=True)
                        return True
                elif rec.get("status", {}).get("error"):
                    print(f"  [{tag}] ERR {rec['status']}", flush=True)
                    return False
        except Exception:
            pass
        if i % 4 == 0:
            print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT", flush=True)
    return False


def main():
    out_base = PROJECT_ROOT / "jobs" / f"smoke_v116_{int(time.time())}"
    out_base.mkdir(parents=True, exist_ok=True)
    print("=== v116 img2img 锁构图 + IPAdapter 锁配色 + 裂变加强（denoise 0.70/0.74）===", flush=True)
    run_one(100301, 0.70, out_base, "d070")
    run_one(100401, 0.74, out_base, "d074")
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
