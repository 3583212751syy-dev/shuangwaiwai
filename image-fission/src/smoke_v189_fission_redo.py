"""
v189 — 修 v188 的"打碎徽章/狗牌"问题

v188 错误：denoise 0.50 + IPA 0.70 → 把 bat_logo 紫色圆形徽章打碎、camo_armed 狗牌打碎成吊牌碎片
v188 优点：LAB 色彩迁移后置兜底配色（保留）

v189 修正方案：
  - 回到 v147 保结构基线：denoise 0.70 / IPA 0.30 (style transfer) / Canny 0.50 / Tile 0.45 / LORA 1.0
  - 双 KSampler 24+20（与 v147 一致）
  - 后置 LAB 色彩迁移保留（防止配色漂移）
  - 提示词明确"PRESERVE emblem/dog tag structure, replace small accents only"
  - 烧字全部走 PIL（与 v189_burn.py 配套）

只跑 bat_logo + camo_armed 两张。
"""

import os, sys, json, time, uuid
from pathlib import Path
import numpy as np
import urllib.request
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v189"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 700510  # 与 v188 不同

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
    "new colors not in original palette, different color palette, extra colors, color shift, "
    "duplicate, watermark"
)

COHESIVE = (
    "cohesive with the rest of the design, natural overlap hierarchy, "
    "fits the overall composition, same art style as the original, "
    "preserve the overall structure and layout of the original emblem/badge"
)

REFS = [
    # === bat_logo: 保留紫色圆形徽章 + 黑蝙蝠 + 粉紫底；改小元素 ===
    {
        "id": "bat_logo",
        "ref_img": "test_6978fabda2cc99629fa9e81f802762d3.jpg",
        "denoise": 0.70,
        "canny_strength": 0.50,
        "tile_strength": 0.45,
        "ipa_weight": 0.30,
        "lora_detail": 1.0,
        "global_pos": (
            "purple circular emblem badge with a central black bat silhouette, "
            "purple and dark purple and black and pink color blocks, "
            "emblem badge art style with circular ring border, "
            "PRESERVE the overall circular emblem composition and bat silhouette, "
            "no readable text, sharp clean edges"
        ),
        "regions": [
            # 主体：保留蝙蝠+徽章圆形结构，只改小细节
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "KEEP the original purple circular badge structure and the central black bat silhouette, "
                 "slightly rotate the bat 15 degrees and enlarge its wingspan to 1.10x, "
                 "replace the small star accents on the badge ring with small triangle accents, "
                 "add 2 more triangle accents around the badge ring (increasing from 2 to 4 accent pieces), "
                 "keep purple/black/pink palette and emblem badge art style. " + COHESIVE)},
        ],
    },

    # === camo_armed: 保留银色狗牌 + 灰色迷彩底；改小元素 ===
    {
        "id": "camo_armed",
        "ref_img": "test_b78e60de8dfdf44acda99395326a7298.jpg",
        "denoise": 0.70,
        "canny_strength": 0.50,
        "tile_strength": 0.45,
        "ipa_weight": 0.30,
        "lora_detail": 1.0,
        "global_pos": (
            "military athletic t-shirt design with dog tag pendant, "
            "pure white background with light gray camouflage texture, "
            "silver metal dog tag with chain, bold stencil typography, "
            "PRESERVE the dog tag pendant and chain composition, "
            "no readable text, sharp clean edges"
        ),
        "regions": [
            # 主体：保留狗牌结构，只改小元素
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "KEEP the original silver dog tag pendant and chain composition, "
                 "slightly rotate the dog tag pendant 12 degrees, "
                 "replace the small star accents scattered around with small eagle silhouette accents, "
                 "change the chain link count slightly (e.g. +2 links), "
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
    for i, r in enumerate(ref["regions"]):
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
        "images": ["14", 0], "filename_prefix": f"v189_{ref['id']}"}}
    return g


def color_transfer(src_rgb, dst_rgb, alpha=1.0):
    """src/dst are RGB numpy. Output RGB."""
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


def hist_intersection(src_rgb, dst_rgb, bins=32):
    import cv2
    inter = 0.0
    for c in range(3):
        hs = cv2.calcHist([src_rgb], [c], None, [bins], [0, 256])
        hd = cv2.calcHist([dst_rgb], [c], None, [bins], [0, 256])
        cv2.normalize(hs, hs); cv2.normalize(hd, hd)
        inter += cv2.compareHist(hs, hd, cv2.HISTCMP_INTERSECT)
    return inter / 3.0


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
        cands = sorted(out_dir.glob(f"v189_{ref['id']}*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        print(f"  ERROR no output for {ref['id']}")
        return None

    src_rgb = np.array(Image.open(COMFY_INPUT / ref["ref_img"]).convert("RGB"))
    dst_rgb = np.array(Image.open(raw_path).convert("RGB"))
    # 后置 LAB 色彩迁移
    matched = color_transfer(src_rgb, dst_rgb, alpha=1.0)
    out_final = JOB / f"v189_{ref['id']}.jpg"
    out_raw = JOB / f"v189_{ref['id']}_raw.jpg"
    Image.fromarray(dst_rgb).save(str(out_raw), quality=95)
    Image.fromarray(matched).save(str(out_final), quality=95)
    hi_before = hist_intersection(src_rgb, dst_rgb)
    hi_after = hist_intersection(src_rgb, matched)
    # 结构差异
    h = min(src_rgb.shape[0], matched.shape[0])
    w = min(src_rgb.shape[1], matched.shape[1])
    s = np.array(Image.fromarray(src_rgb).resize((w, h)))
    d = np.array(Image.fromarray(matched).resize((w, h)))
    mse = ((s.astype(np.float32) - d.astype(np.float32)) ** 2).mean()
    sd = float(np.clip(1 - mse / (255.0 ** 2), 0, 1))
    print(f"  saved v189_{ref['id']}.jpg  ({out_final.stat().st_size//1024} KB)")
    print(f"    配色交集: 迁移前={hi_before:.3f} → 迁移后={hi_after:.3f}  | 结构差异(裂变度)={sd:.3f}")
    if hi_after < 0.80:
        print(f"    ⚠ 配色交集仍偏低({hi_after:.3f})")
    if sd < 0.40:
        print(f"    ⚠ 结构差异过低({sd:.3f})，可能没真裂变")
    if sd > 0.85:
        print(f"    ⚠ 结构差异过高({sd:.3f})，可能崩了")
    return str(out_final)


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    for r in REFS:
        if r["id"] not in targets:
            continue
        out = JOB / f"v189_{r['id']}.jpg"
        if out.exists() and out.stat().st_size > 100000:
            print(f"[skip] {r['id']} already done")
            continue
        gen(r, SEED)


if __name__ == "__main__":
    main()