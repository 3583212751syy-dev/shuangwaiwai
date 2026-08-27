"""v121 终版：按图分档 IPAdapter 强度 + 禁字防乱字 + 严格关联原图
用户规则（2026-08-27 确立）：
- 纯色图 → IPAdapter 0.3-0.45（illust_1 黑白 / denim_3 靛蓝白 → 0.45）
- 复杂图 → IPAdapter 0.6-0.8（eagle_2 / camo_4 / skull_5 / metal_6 → 0.70）
- 严格要求跟参考图有关联：主体/构图/配色/风格保留，只改侵权元素和文字
- AI 生成时禁止任何文字（防乱字糊字）→ 文字全部走 PIL 后期烧字（burn_text.py）
- 单词有意义：denim_3=UPCY / eagle_2=FERAL / skull_5=VENOM / metal_6=THRASH

用法：python src/smoke_v121_final.py
"""
import time, requests, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REFS = [
    {   # 纯色 → 0.45
        "id": "illust_1", "ref_img": "pinterest_illust_1.jpg", "weight": 0.45,
        "subject": "elegant peacock silhouette with elaborate flowing tail feathers, baroque vector style",
        "palette": "pure black background, white linework only",
    },
    {   # 纯色 → 0.45
        "id": "denim_3", "ref_img": "pinterest_denim_3.jpg", "weight": 0.45,
        "subject": "single delicate moth flying over a distressed X-shaped denim patch",
        "palette": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark, fabric wear and tear, dotted flight trail",
    },
    {   # 复杂 → 0.70
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": 0.70,
        "subject": "majestic spread-winged eagle gripping a skull in its talons, gothic crest emblem with chains and flames",
        "palette": "dark black background, white and red linework, intricate feathers, heraldic crest shield",
    },
    {   # 复杂 → 0.70
        "id": "camo_4", "ref_img": "pinterest_camo_4.jpg", "weight": 0.70,
        "subject": "tropical palm tree silhouette with classic military camouflage pattern",
        "palette": "tropical jungle camouflage, brown green tan khaki black palette, organic camo blob shapes",
    },
    {   # 复杂 → 0.70
        "id": "skull_5", "ref_img": "pinterest_skull_5.jpg", "weight": 0.70,
        "subject": "grim human skull with red snake coiled around it and red angel wings",
        "palette": "gothic tattoo illustration, pure black background, red and white palette, blood drips, serpent scales, feathered wings",
    },
    {   # 复杂 → 0.70
        "id": "metal_6", "ref_img": "pinterest_metal_6.jpg", "weight": 0.70,
        "subject": "bald eagle perched on a cracked human skull with horn-like spikes and lightning bolts",
        "palette": "death metal illustration, pure black background, white and bronze gold yellow palette, radial lightning burst, heavy ink",
    },
]

POS_TAIL = (
    "same composition and layout as reference image, same art style and color palette as reference, "
    "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
    "fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain"
)

NEG = (
    "text, letters, words, typography, captions, watermark, logo, brand name, "
    "no gray, no midtones, no gradient, no soft shading, no halftone, "
    "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, "
    "margin, border, frame, padding, empty corners, letterbox, white border, smudge, "
    "duplicate image, exact copy"
)


def build(ref, seed):
    weight = ref["weight"]
    pos = (f"t-shirt graphic design, {ref['subject']}, {ref['palette']}, "
           f"{POS_TAIL}")
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
        "weight": weight, "weight_type": "style transfer",
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
                          "filename_prefix": f"v121_{ref['id']}_w{int(weight*100):03d}"}}
    return g


def run(ref, seed, out_base):
    tag = f"{ref['id']}_w{int(ref['weight']*100):03d}"
    r = requests.post(f"{COMFYUI}/prompt",
                      json={"prompt": build(ref, seed),
                            "client_id": f"v121_{int(time.time())}"}, timeout=10)
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
    out = PROJECT_ROOT / "jobs" / f"smoke_v121_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print("=== v121 终版 6 张（纯色0.45/复杂0.70，禁字，严格关联原图）===", flush=True)
    seed = 300301
    for ref in REFS:
        run(ref, seed, out); seed += 100
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())