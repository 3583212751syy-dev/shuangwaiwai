"""
v190 — 保守参数重做 bat_logo + camo_armed

v189 错误：
  - bat_logo：denoise 0.70 + IPA 0.30 + Canny 0.50 把徽章圆环削了
  - camo_armed：denoise 0.70 + IPA 0.30 把狗牌过大占满 + 背景变深灰 + 内部加鹰形乱

v190 修正：
  - denoise 0.55（中段，给裂变留空间但不破坏结构）
  - IPA 0.40（强锁色，靠 LAB 后置兜底）
  - Canny 0.60（强化锁轮廓）
  - Tile 0.50
  - LORA 1.0
  - 提示词更严格：bat_logo 强化圆环，camo_armed 强调"狗牌小/居中/背景浅灰"
"""

import os, sys, json, time, uuid
from pathlib import Path
import numpy as np
import urllib.request
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v190"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 700520

NEG_BASE = (
    "frame, border, white border, edge border, letterbox, "
    "text, letters, words, writing, typography, signature, caption, "
    "garbled text, illegible letters, mangled typography, distorted characters, broken text, "
    "wrong letters, scrambled letters, partially missing letters, "
    "3d, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, watercolor, pastel, "
    "noise, grain, pixelated, jagged edges, aliasing, "
    "mutated, malformed, deformed anatomy, extra limbs, "
    "melted, fused, smudged, bleeding, "
    "dark gray, dark background, dark color, "
    "new colors not in original palette, different color palette, extra colors, color shift, "
    "oversized, zoomed in too much, filling entire frame, "
    "duplicate, watermark"
)

COHESIVE = (
    "cohesive with the rest of the design, natural overlap hierarchy, "
    "fits the overall composition, same art style as the original, "
    "preserve the overall structure and layout"
)

REFS = [
    # === bat_logo: 强化圆环 + 三角加在徽章外环 ===
    {
        "id": "bat_logo",
        "ref_img": "test_6978fabda2cc99629fa9e81f802762d3.jpg",
        "denoise": 0.55,
        "canny_strength": 0.60,
        "tile_strength": 0.50,
        "ipa_weight": 0.40,
        "lora_detail": 1.0,
        "global_pos": (
            "purple circular emblem badge with bat silhouette and outer ring border, "
            "purple and dark purple and black and pink color blocks, "
            "emblem badge art style with full circular ring border and arc text zone, "
            "PRESERVE the entire circular badge including outer ring, arc text zone, "
            "and the inner central bat silhouette composition, "
            "no readable text, sharp clean edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "STRICTLY PRESERVE the entire circular badge outer ring and the inner bat silhouette, "
                 "slightly rotate the bat 8 degrees and enlarge its wingspan to 1.08x (small change only), "
                 "add 2 small triangle accents ON the outer ring border near the top corners "
                 "(instead of the original star accents), "
                 "keep purple/black/pink palette and emblem badge art style. " + COHESIVE)},
        ],
    },

    # === camo_armed: 狗牌保持小 + 背景浅灰不加鹰形 ===
    {
        "id": "camo_armed",
        "ref_img": "test_b78e60de8dfdf44acda99395326a7298.jpg",
        "denoise": 0.55,
        "canny_strength": 0.60,
        "tile_strength": 0.50,
        "ipa_weight": 0.40,
        "lora_detail": 1.0,
        "global_pos": (
            "white background with light gray camouflage texture, "
            "small silver metal dog tag pendant with chain in the center, "
            "clean product print mockup style with bright white background, "
            "KEEP the dog tag SMALL (about 1/3 of image height) and centered, "
            "KEEP the background LIGHT (bright white with light gray camouflage), NOT dark, "
            "no readable text, sharp clean edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "KEEP the dog tag pendant SMALL (about 1/3 of image height) and centered, "
                 "slightly rotate the dog tag pendant 8 degrees (small change only), "
                 "KEEP the silver metal appearance and chain, "
                 "KEEP the background BRIGHT WHITE with LIGHT GRAY camouflage texture (NOT dark gray), "
                 "do NOT add any eagle or extra decoration inside the dog tag, "
                 "keep pure white/light gray/black palette and military stencil style. " + COHESIVE)},
        ],
    },
]


def build(ref, seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {
        "model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": ref["ipa_weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": ref["lora_detail"], "strength_clip": ref["lora_detail"]}}
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": ref["canny_strength"]}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": ref["tile_strength"]}}

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": ref["global_pos"]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    for i, r in enumerate(ref["regions"]):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"]}}

    comb_in = {"global_cond": ["pg", 0]}
    for i, ref_item in enumerate(ref["regions"]):
        comb_in[f"region{i+1}"] = [f"sa{i}", 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0],
        "seed": seed, "steps": 24, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": ref["denoise"]}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0],
        "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {
        "images": ["14", 0], "filename_prefix": f"v190_{ref['id']}"}}
    return g


def color_transfer(src_rgb, dst_rgb, alpha=1.0):
    import cv2
    src = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = dst.copy()
    for i in range(3):
        s_mean, s_std = src[:, :, i].mean(), src[:, :, i].std() + 1e-6
        d_mean, d_std = dst[:, :, i].mean(), dst[:, :, i].std() + 1e-6
        out[:, :, i] = (dst[:, :, i] - d_mean) * (s_std / d_std) + s_mean
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    if alpha >= 1.0:
        return out
    blended = dst.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())


def gen(ref, seed):
    g = build(ref, seed)
    r = submit(g)
    pid = r["prompt_id"]
    print(f"  submitted {ref['id']} pid={pid[:8]} denoise={ref['denoise']} ipa={ref['ipa_weight']}")
    raw_path = None
    for _ in range(96):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print(f"  TIMEOUT {ref['id']}")
        return None
    outputs = h[pid].get("outputs", {})
    for node_id, info in outputs.items():
        for img in info.get("images", []):
            src = Path("E:/Desktop/双接口/image-fission/ComfyUI/output") / img["filename"]
            if src.exists():
                raw_path = str(src)
                break
        if raw_path:
            break
    if not raw_path:
        out_dir = Path("E:/Desktop/双接口/image-fission/ComfyUI/output")
        cands = sorted(out_dir.glob(f"v190_{ref['id']}*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        print(f"  ERROR no output for {ref['id']}")
        return None

    src_rgb = np.array(Image.open(COMFY_INPUT / ref["ref_img"]).convert("RGB"))
    dst_rgb = np.array(Image.open(raw_path).convert("RGB"))
    matched = color_transfer(src_rgb, dst_rgb, alpha=1.0)
    out_final = JOB / f"v190_{ref['id']}.jpg"
    out_raw = JOB / f"v190_{ref['id']}_raw.jpg"
    Image.fromarray(dst_rgb).save(str(out_raw), quality=95)
    Image.fromarray(matched).save(str(out_final), quality=95)
    print(f"  saved v190_{ref['id']}.jpg  ({out_final.stat().st_size//1024} KB)")
    return str(out_final)


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    for r in REFS:
        if r["id"] not in targets:
            continue
        out = JOB / f"v190_{r['id']}.jpg"
        if out.exists() and out.stat().st_size > 100000:
            print(f"[skip] {r['id']} already done")
            continue
        gen(r, SEED)


if __name__ == "__main__":
    main()