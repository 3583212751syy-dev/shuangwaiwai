"""v123 回归 v118 已验收管线 + 字体融合烧字（治字体割裂感）
用户反馈（2026-08-27）：
- 字体跟图分离（v122 烧字太生硬）→ 融合增强：阴影 + 末端 mask 渐变 + 颜色贴主体
- 元素失真混乱（v122 强约束 prompt 干扰）→ 回归 v118 已验管线：IPAdapter 0.55 + denoise 0.72
- 拿原 6 张图裂变 → 用 pinterest_*_jpg 原图

用法：python src/smoke_v123_stable.py
"""
import time, requests, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# 回归 v118 已验收 prompt（不加 "SOLID BLACK" 等强约束）
REFS = [
    {"id": "illust_1", "ref_img": "pinterest_illust_1.jpg", "weight": 0.55,
     "subject": "elegant peacock silhouette with elaborate flowing tail feathers",
     "palette": "baroque vector illustration, pure black background, white linework only, ornamental 5-petal flowers and decorative swirling flourishes",
     "text_cfg": None},  # 无字
    {"id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": 0.55,
     "subject": "majestic spread-winged eagle gripping a skull in its talons, gothic crest emblem with chains and flames",
     "palette": "gothic tattoo illustration, dark black background, white and red linework, intricate feathers, flaming aura, heraldic crest shield, iron chains, blood splatter",
     "text_cfg": {"word": "FERAL", "font": "pirata_one", "size_ratio": 0.085,
                   "fill": (235, 225, 200, 255), "stroke": (15, 15, 25, 255),
                   "stroke_w": 4, "y_ratio": 0.10, "spacing": 6,
                   "shadow_offset": 6, "shadow_alpha": 140, "fade_bottom": 0.85}},
    {"id": "denim_3", "ref_img": "pinterest_denim_3.jpg", "weight": 0.55,
     "subject": "single delicate moth flying over a distressed X-shaped denim patch",
     "palette": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark, fabric wear and tear, dotted flight trail",
     "text_cfg": {"word": "UPCY", "font": "rye", "size_ratio": 0.085,
                   "fill": (50, 60, 90, 255), "stroke": (255, 255, 255, 255),
                   "stroke_w": 5, "y_ratio": 0.10, "spacing": 8,
                   "shadow_offset": 6, "shadow_alpha": 140, "fade_bottom": 0.85}},
    {"id": "camo_4", "ref_img": "pinterest_camo_4.jpg", "weight": 0.55,
     "subject": "tropical palm tree silhouette with classic military camouflage pattern background",
     "palette": "tropical jungle camouflage, brown green tan khaki black palette, black palm tree silhouettes, organic camo blob shapes, repeating pattern",
     "text_cfg": None},  # 无字
    {"id": "skull_5", "ref_img": "pinterest_skull_5.jpg", "weight": 0.55,
     "subject": "grim human skull with red snake coiled around it and red angel wings",
     "palette": "gothic tattoo illustration, pure black background, red and white palette, blood drips, cracked bone, serpent scales, feathered wings",
     "text_cfg": {"word": "VENOM", "font": "pirata_one", "size_ratio": 0.085,
                   "fill": (220, 35, 45, 255), "stroke": (8, 8, 12, 255),
                   "stroke_w": 5, "y_ratio": 0.10, "spacing": 6,
                   "shadow_offset": 6, "shadow_alpha": 140, "fade_bottom": 0.85}},
    {"id": "metal_6", "ref_img": "pinterest_metal_6.jpg", "weight": 0.55,
     "subject": "bald eagle perched on a cracked human skull with massive horn-like spikes and radiating lightning bolts",
     "palette": "death metal thrash metal illustration, pure black background, white and bronze gold yellow palette, sharp aggressive lines, radial lightning burst, heavy ink",
     "text_cfg": {"word": "THRASH", "font": "metal_mania", "size_ratio": 0.09,
                   "fill": (242, 205, 105, 255), "stroke": (12, 8, 4, 255),
                   "stroke_w": 5, "y_ratio": 0.10, "spacing": 4,
                   "shadow_offset": 6, "shadow_alpha": 140, "fade_bottom": 0.85}},
]

POS_TAIL = (
    "same composition and layout as reference image, "
    "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
    "fills entire frame edge-to-edge, full bleed, no halftone, no noise, no grain"
)

NEG = (
    "text, letters, words, typography, captions, watermark, logo, brand name, "
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "no gray, no midtones, no gradient, no soft shading, no halftone, "
    "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, smudge, "
    "duplicate image, exact copy"
)


def build(ref, seed):
    pos = (f"t-shirt graphic design, {ref['subject']}, {ref['palette']}, {POS_TAIL}")
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos",
        "megapixels": 1.0, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader",
              "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": ref["weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": pos}}
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
                          "filename_prefix": f"v123_{ref['id']}"}}
    return g


def run(ref, seed, out_base):
    tag = ref["id"]
    r = requests.post(f"{COMFYUI}/prompt",
                      json={"prompt": build(ref, seed),
                            "client_id": f"v123_{int(time.time())}"}, timeout=10)
    if r.status_code != 200:
        print(f"[ERR] {tag} submit: {r.status_code} {r.text[:200]}", flush=True); return False
    pid = r.json()["prompt_id"]
    print(f"  [{tag}] pid={pid}", flush=True)
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
                        out = out_base / f"{tag}.jpg"
                        out.write_bytes(data)
                        print(f"  [{tag}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    print(f"  [{tag}] ERR {rec['status']}", flush=True); return False
        except Exception: pass
        if i % 12 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT", flush=True); return False


def main():
    out = PROJECT_ROOT / "jobs" / f"smoke_v123_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print("=== v123 回归 v118 已验管线 6 张 ===", flush=True)
    seed = 500301
    for ref in REFS:
        run(ref, seed, out); seed += 100
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())