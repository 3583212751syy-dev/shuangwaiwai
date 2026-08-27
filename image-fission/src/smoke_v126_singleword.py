"""v126 优化 Harrlogos：多词改单字 + 减修饰词 + 每张 3 seed 拾优
用户反馈（2026-08-27）：v125 Harrlogos 多词拼写错（JACKE DIANNIES→JACKONES）、装饰过重。
策略（用户拍板「优化 Harrlogos 多词改单字」）：
- 多词全部改单字有意义词（Harrlogos 官方：单字最稳）：
  eagle_2: JACKE DIANNIES → DOMINION (7字符, 死亡金属气场)
  skull_5: TRUE NEVER DIES → VENOM (5字符, 蛇毒主题)
  denim_3: UPCY (4字符单字, 保留)
  metal_6: MRCHGSR (7字符单字, 保留)
- 减修饰词：去掉 dripping/splattered/stitched/fire/flames 等"装饰糊字"的 modifier，
  只留颜色 + 核心风格词（metal/gothic/vintage/denim）
- 每张 3 个 seed（Harrlogos 拼写随 seed 变），present 后用户拾优
- 保持 v125 已验证管线：img2img 锁构图 + IPAdapter 0.55 锁画风 + Harrlogos 0.8 画字

用法：python src/smoke_v126_singleword.py
"""
import time, requests, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REFS = [
    {"id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": 0.55,
     "subject": ("double-headed eagle crest emblem with flaming aura and iron chains, "
                 "skull throne at base, heraldic shield with banner"),
     "palette": "gothic tattoo illustration, dark black background, white and red linework, intricate feathers",
     "word": "DOMINION",
     "text_prompt": '("DOMINION":1.4) text logo, white, grey, black, metal, spikey, gothic',
     "extra_neg": "Double letters, repeating letters, malformed letters, more than 8 letters, typo, misspelled"},
    {"id": "denim_3", "ref_img": "pinterest_denim_3.jpg", "weight": 0.55,
     "subject": "delicate moth flying over a distressed X-shaped denim patch with fabric wear",
     "palette": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark",
     "word": "UPCY",
     "text_prompt": '("UPCY":1.5) text logo, blue, white, denim, vintage',
     "extra_neg": "Double letters, repeating letters, malformed letters, more than 4 letters, typo, misspelled"},
    {"id": "skull_5", "ref_img": "pinterest_skull_5.jpg", "weight": 0.55,
     "subject": "grim human skull with black eye patch, red snake coiled around it, red angel wings, red rose",
     "palette": "gothic tattoo illustration, pure black background, red and white palette",
     "word": "VENOM",
     "text_prompt": '("VENOM":1.4) text logo, red, white, black, gothic',
     "extra_neg": "Double letters, repeating letters, malformed letters, more than 5 letters, typo, misspelled"},
    {"id": "metal_6", "ref_img": "pinterest_metal_6.jpg", "weight": 0.55,
     "subject": "bald eagle perched on cracked human skull with massive horn-like spikes and radiating lightning bolts",
     "palette": "death metal thrash metal illustration, pure black background, white and bronze gold yellow palette",
     "word": "MRCHGSR",
     "text_prompt": '("MRCHGSR":1.4) text logo, white, grey, black, metal, spikey',
     "extra_neg": "Double letters, repeating letters, malformed letters, more than 7 letters, typo, misspelled"},
]

POS_TAIL = (
    "same composition and layout as reference image, "
    "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
    "fills entire frame edge-to-edge, full bleed, no halftone, no noise, no grain"
)

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "no gray, no midtones, no gradient, no soft shading, no halftone, "
    "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, smudge, "
    "duplicate image, exact copy, watermark"
)

SEEDS = [700301, 700302, 700303]


def build(ref, seed):
    pos = (f"t-shirt graphic design, {ref['subject']}, {ref['palette']}, "
           f"{ref['text_prompt']}, {POS_TAIL}")
    neg = f"{NEG_BASE}, {ref['extra_neg']}"
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
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": "Harrlogos_XL_v2.safetensors",
        "strength_model": 0.8, "strength_clip": 0.8}}
    g["8"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["7", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["7", 1], "text": neg}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 32, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.72}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 28, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage",
               "inputs": {"images": ["14", 0],
                          "filename_prefix": f"v126_{ref['id']}_s{seed}"}}
    return g


def run(ref, seed, out_base):
    tag = f"{ref['id']}_s{seed}"
    r = requests.post(f"{COMFYUI}/prompt",
                      json={"prompt": build(ref, seed),
                            "client_id": f"v126_{int(time.time())}"}, timeout=10)
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
                    imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
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
    out = PROJECT_ROOT / "jobs" / f"smoke_v126_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print("=== v126 单字 Harrlogos（4 图 × 3 seed = 12 张）===", flush=True)
    for ref in REFS:
        for s in SEEDS:
            run(ref, s, out)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())