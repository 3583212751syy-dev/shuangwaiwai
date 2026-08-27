"""v119 IPAdapter 强度预览：illust_1 跑 4 档 weight (0.30/0.45/0.60/0.80)
给前端 slider 切换用——展示 IPAdapter 强度对画面的影响（配色继承、风格锁松紧）。
其余参数沿用 v117 已验证：img2img 原图比例 + denoise 0.72 + REDESIGNED prompt。

用法：python src/smoke_v119_strength_preview.py
"""
import time, requests, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REF_IMG = "pinterest_illust_1.jpg"
POS = (
    "t-shirt graphic design, elegant peacock silhouette with elaborate flowing tail feathers, "
    "same composition and layout as reference image, pure black background, white linework only, "
    "high contrast, "
    "REDESIGNED internal details, new intricate ornamental patterns, "
    "beautifully reimagined feather textures, refined artistic variations, "
    "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
    "fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain"
)
NEG = (
    "no gray, no midtones, no color, no gradient, no soft shading, no halftone, "
    "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, "
    "margin, border, frame, padding, empty corners, letterbox, white border, smudge, "
    "text, letters, watermark, duplicate image, exact copy"
)
WEIGHTS = [0.30, 0.45, 0.60, 0.80]


def build(seed, weight):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMG}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos",
        "megapixels": 1.0, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader",
              "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": weight, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": POS}}
    g["8"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": NEG}}
    g["9"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0], "positive": ["7", 0], "negative": ["8", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 45, "cfg": 7.5,
        "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.72}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0], "positive": ["7", 0], "negative": ["8", 0],
        "latent_image": ["9", 0], "seed": seed + 1, "steps": 40, "cfg": 7.5,
        "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.20}}
    g["11"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["13"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["12", 0], "image": ["11", 0]}}
    g["14"] = {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0],
                          "filename_prefix": f"v119_w{int(weight*100):03d}"}}
    return g


def run(seed, weight, out_base):
    w = requests.post(f"{COMFYUI}/prompt",
                      json={"prompt": build(seed, weight),
                            "client_id": f"v119_{int(time.time())}"}, timeout=10)
    if w.status_code != 200:
        print(f"[ERR] w{weight} submit failed: {w.status_code}", flush=True); return False
    pid = w.json()["prompt_id"]
    print(f"  [w{weight}] pid={pid}", flush=True)
    for i in range(150):
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=5).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("14", {}).get("images", [])
                    if imgs:
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        data = requests.get(url, timeout=30).content
                        out = out_base / f"w{int(weight*100):03d}.jpg"
                        out.write_bytes(data)
                        print(f"  [w{weight}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    print(f"  [w{weight}] ERR {rec['status']}", flush=True); return False
        except Exception: pass
        if i % 12 == 0: print(f"    [w{weight}] {i*5}s...", flush=True)
    print(f"  [w{weight}] TIMEOUT", flush=True); return False


def main():
    out = PROJECT_ROOT / "jobs" / f"smoke_v119_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print("=== v119 IPAdapter 强度预览 4 档 (illust_1) ===", flush=True)
    seed = 100301
    for w in WEIGHTS:
        run(seed, w, out); seed += 100
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())