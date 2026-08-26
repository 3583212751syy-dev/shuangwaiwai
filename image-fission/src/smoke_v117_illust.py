"""v117 尺寸保真版：不改变原图尺寸/比例（用户要求）
用户反馈（2026-08-26）：「不加多余颜色不改变原图尺寸就很好了」
- 不加多余颜色：IPAdapter style 0.55 锁黑底白线（沿用 v116，已达标）
- 不改变原图尺寸：原图是 558×960 竖图，但 enhance 时被裁成 4096² 正方形 → 改用原始竖图
  + ImageScaleToTotalPixels(lanczos, 1.0MP) 保持宽高比缩放 → latent 竖构图 → 输出竖图
- 裂变程度：沿用 v116 denoise 0.72（用户认可的裂变强度）

用法：python src/smoke_v117_illust.py
"""
import time, requests, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REF_IMG = "pinterest_illust_1.jpg"   # 原始竖图 558×960（不用被裁方的 HD 图）

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
    """v117：原始竖图 -> 保持比例缩放 -> img2img 锁竖构图 + IPAdapter 锁配色 + 裂变 0.72
    1: CheckpointLoader
    2: LoadImage (原始竖图)
    3: ImageScaleToTotalPixels (lanczos, 1.0MP，保持宽高比)
    4: VAEEncode
    5: IPAdapterUnifiedLoader
    6: IPAdapterAdvanced (weight=0.55, style transfer)
    7: CLIPTextEncode pos
    8: CLIPTextEncode neg
    9: KSampler 1 (denoise=0.72)
    10: KSampler 2 (denoise=0.20)
    11: VAEDecode
    12: UpscaleModelLoader
    13: ImageUpscaleWithModel
    14: SaveImage
    """
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMG}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0],
        "upscale_method": "lanczos",
        "megapixels": 1.0,
        "resolution_steps": 64,
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
               "inputs": {"images": ["13", 0], "filename_prefix": "v117_illust_1"}}
    return g


def main():
    out_base = PROJECT_ROOT / "jobs" / f"smoke_v117_{int(time.time())}"
    out_base.mkdir(parents=True, exist_ok=True)

    seed = 100301
    denoise = 0.72
    print(f"=== v117 尺寸保真版：illust_1（竖图 {REF_IMG}，denoise {denoise}）===", flush=True)

    workflow = build_workflow(seed, denoise)
    payload = {"prompt": workflow, "client_id": f"v117_{int(time.time())}"}
    r = requests.post(f"{COMFYUI}/prompt", json=payload, timeout=10)
    if r.status_code != 200:
        print(f"[ERR] submit failed: {r.status_code} {r.text[:300]}", flush=True)
        return 1
    pid = r.json().get("prompt_id")
    print(f"  prompt_id={pid}", flush=True)

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
                        out = out_base / f"illust_1_{seed}_d{str(denoise).replace('.', '')}.jpg"
                        out.write_bytes(data)
                        print(f"  [OK] saved {out} ({out.stat().st_size/1024/1024:.1f}MB)", flush=True)
                        return 0
                elif rec.get("status", {}).get("error"):
                    print(f"  [ERR] error: {rec['status']}", flush=True)
                    return 2
        except Exception:
            pass
        if i % 4 == 0:
            print(f"    {i*5}s...", flush=True)

    print("  [TIMEOUT] 600s", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
