"""v124 用用户指定的 6 张原图（Eagle库/clipboard）重做裂变
v123 之前的 6 张图里 eagle_2 / metal_6 命名/内容错误（comf 里是相反的）。
用户 2026-08-27 给出的真实 6 张原图：
1. illust_1: 黑白卷草花饰（巴洛克风，无字）— ComfyUI 原版已对
2. denim_3: UPCY 牛仔布字母 + 牛仔布蝴蝶 — ComfyUI 原版已对
3. camo_4: 棕榈树迷彩 — ComfyUI 原版已对
4. eagle_2: 双鹰王座 + JACKIE DIANNIES 火焰盾牌 + 骷髅 + 锁链
5. skull_5: 骷髅蛇翼 + TRUE NEVER DIES 红字 + 玫瑰血滴 — ComfyUI 原版已对
6. metal_6: 死亡金属鹰骷髅 + MRCHGSR 白色 logo + 闪电 + 角刺

按原图真词烧字（不再用占位词 FERAL/VENOM/THRASH）：
- eagle_2: JACKIE DIANNIES (PirataOne 哥特, 火焰灰白)
- skull_5: TRUE NEVER DIES (PirataOne 哥特, 红)
- metal_6: MRCHGSR (MetalMania 死亡金属, 白色不是金黄!)
- denim_3: UPCY (Rye 牛仔)
"""
import time, requests, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# 用户给的真实 6 张原图路径
REFS = [
    {"id": "illust_1", "ref_img": "pinterest_illust_1.jpg", "weight": 0.55,
     "subject": "baroque ornamental floral arrangement with elegant curling acanthus leaves and 5-petal flowers",
     "palette": "baroque vector illustration, pure black background, white linework only, ornamental 5-petal flowers, swirling flourishes"},
    {"id": "denim_3", "ref_img": "pinterest_denim_3.jpg", "weight": 0.55,
     "subject": "delicate moth flying over a distressed X-shaped denim patch with fabric wear",
     "palette": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark, dotted flight trail"},
    {"id": "camo_4", "ref_img": "pinterest_camo_4.jpg", "weight": 0.55,
     "subject": "tropical palm tree silhouette with classic military camouflage pattern",
     "palette": "tropical jungle camouflage, brown green tan khaki black palette, black palm tree silhouettes, organic camo blob shapes"},
    {"id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": 0.55,
     "subject": "double-headed eagle crest emblem with flaming aura and iron chains, skull throne at base, heraldic shield with banner",
     "palette": "gothic tattoo illustration, dark black background, white and red linework, intricate feathers, flaming aura, blood splatter, chained shield"},
    {"id": "skull_5", "ref_img": "pinterest_skull_5.jpg", "weight": 0.55,
     "subject": "grim human skull with black eye patch, red snake coiled around it, red angel wings, red rose, dripping blood",
     "palette": "gothic tattoo illustration, pure black background, red and white palette, blood drips, serpent scales, feathered wings"},
    {"id": "metal_6", "ref_img": "pinterest_metal_6.jpg", "weight": 0.55,
     "subject": "bald eagle perched on cracked human skull with massive horn-like spikes and radiating lightning bolts",
     "palette": "death metal thrash metal illustration, pure black background, white and bronze gold yellow palette, sharp aggressive lines, radial lightning burst, heavy ink"},
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
    pos = f"t-shirt graphic design, {ref['subject']}, {ref['palette']}, {POS_TAIL}"
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
                          "filename_prefix": f"v124_{ref['id']}"}}
    return g


def run(ref, seed, out_base):
    tag = ref["id"]
    r = requests.post(f"{COMFYUI}/prompt",
                      json={"prompt": build(ref, seed),
                            "client_id": f"v124_{int(time.time())}"}, timeout=10)
    if r.status_code != 200:
        print(f"[ERR] {tag}: {r.status_code} {r.text[:200]}", flush=True); return False
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
        except: pass
        if i % 12 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT", flush=True); return False


def main():
    out = PROJECT_ROOT / "jobs" / f"smoke_v124_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print("=== v124 用用户真实 6 张原图重做裂变 ===", flush=True)
    seed = 600301
    for ref in REFS:
        run(ref, seed, out); seed += 100
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())