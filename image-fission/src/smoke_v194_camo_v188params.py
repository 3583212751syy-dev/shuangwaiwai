"""
v194 — camo_armed 用 v188 参数重做（字作为图元素方案）

用户 2026-09-01 16:30 拍板：
- bat_logo v193 永久采用（8.5/10 够用）
- camo_armed 重做：改用 v188 参数（IPA 0.70 / denoise 0.50 / Canny 0.50 / Tile 0.60 / LORA 0.40）
  → 锁住白底（T 恤设计稿必须有纯白背景）+ 字作为装饰图形

v193 同提示词，但换参数。后置 Reinhard LAB 色彩迁移兜底。
"""
import os, sys, json, time, shutil, uuid
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# ==================== 配置 ====================
PROJECT = Path(__file__).resolve().parents[1]
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v194"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 700520

# ==================== v188 参数（用户授权调参，仅此一张图）====================
DENOISE = 0.50
IPA_WEIGHT = 0.70
LORA_DETAIL = 0.40  # v188 降噪保字
CANNY_STRENGTH = 0.50
TILE_STRENGTH = 0.60

# ==================== NEG_BASE（v188 标准，无 region，强度更高）====================
NEG_BASE = (
    "frame, border, white border, letterbox, "
    "concrete wall, cracked wall, dirty background, textured wall, gray noise background, "
    "garbled text, illegible letters, mangled typography, distorted characters, broken text, "
    "wrong letters, scrambled letters, partially missing letters, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, watercolor, pastel, "
    "noise, grain, pixelated, jagged edges, aliasing, "
    "mutated, malformed, deformed anatomy, extra limbs, "
    "melted, fused, smudged, bleeding, "
    "new colors not in original palette, different color palette, extra colors, color shift, "
    "duplicate, watermark"
)

COHESIVE = (
    "cohesive with the rest of the design, natural overlap hierarchy, "
    "fits the overall composition, same art style as the original"
)

# ==================== camo_armed REFS（v193 提示词 + v188 参数）====================
REFS = [
    {
        "id": "camo_armed",
        "ref_img": "test_b78e60de8dfdf44acda99395326a7298.jpg",
        "global_pos": (
            "clean white military support athletic t-shirt product design, "
            "pure white #FFFFFF and dark charcoal #2A2A2A and black ONLY color blocks, "
            "a military dog tag pendant centered, "
            "decorative ornamental STYLIZED LETTERING on the dog tag as graphical elements "
            "(looks like abstract letter-shaped patterns engraved on the metal, NOT readable English words, "
            "NOT spelled-out words, NOT brand names), "
            "military stencil art style, clean PRODUCT PHOTOGRAPHY background (PURE WHITE seamless backdrop), "
            "NO concrete wall, NO textured wall, NO gray noise, NO dirty background"
        ),
        "regions": [
            # 主体狗牌：旋转 + 字作为金属上的装饰性艺术字
            {"x": 0.20, "y": 0.25, "w": 0.60, "h": 0.60, "strength": 1.30,
             "prompt": (
                 "a military dog tag pendant centered, rotated 18 degrees clockwise and enlarged to 1.20x, "
                 "decorative ornamental stylized lettering engraved on the dog tag as abstract letter-shaped "
                 "patterns (looks like artistic metal-engraving patterns, NOT readable English words, "
                 "NOT spelled-out words, NOT brand names, NOT legible letters), "
                 "the letterform patterns should look like a vintage military engraving style "
                 "with curves and flourishes that suggest 'words' without spelling any specific word, "
                 "keep pure white background and black military stencil art style. " + COHESIVE)},
            # 顶部徽章带：装饰性图案
            {"x": 0.15, "y": 0.00, "w": 0.70, "h": 0.20, "strength": 1.05,
             "prompt": (
                 "top header band with abstract geometric military ornaments and stylized "
                 "letter-shaped ribbon patterns (NOT readable text, NOT English words), "
                 "pure white background with charcoal and black decorative elements. " + COHESIVE)},
            # 底部：装饰星 + 装饰带
            {"x": 0.15, "y": 0.88, "w": 0.70, "h": 0.12, "strength": 1.00,
             "prompt": (
                 "a thin bottom band with 5 small military star accents and stylized ornamental ribbons "
                 "with abstract letterform patterns woven in (NOT readable text, NOT English words), "
                 "pure white background. " + COHESIVE)},
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

    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}

    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CANNY_STRENGTH}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": TILE_STRENGTH}}

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": ref["global_pos"]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    region_nodes = []
    for i, r in enumerate(ref["regions"]):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        # 用全 1.0 强度（v188 不缩 0.55）
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"]}}
        region_nodes.append(sk)

    comb_in = {"global_cond": ["pg", 0]}
    for i, sk in enumerate(region_nodes):
        comb_in[f"region{i+1}"] = [sk, 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 24, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v194_{ref['id']}"}}
    return g


# ==================== 后置 ====================
def color_transfer(src_bgr, dst_bgr, alpha=1.0):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = dst.copy()
    for i in range(3):
        s_mean, s_std = src[:, :, i].mean(), src[:, :, i].std() + 1e-6
        d_mean, d_std = dst[:, :, i].mean(), dst[:, :, i].std() + 1e-6
        out[:, :, i] = (dst[:, :, i] - d_mean) * (s_std / d_std) + s_mean
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if alpha >= 1.0:
        return out
    blended = dst.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def hist_intersection(src_bgr, dst_bgr, bins=32):
    inter = 0.0
    for c in range(3):
        hs = cv2.calcHist([src_bgr], [c], None, [bins], [0, 256])
        hd = cv2.calcHist([dst_bgr], [c], None, [bins], [0, 256])
        cv2.normalize(hs, hs); cv2.normalize(hd, hd)
        inter += cv2.compareHist(hs, hd, cv2.HISTCMP_INTERSECT)
    return inter / 3.0


def structural_diff(src_bgr, dst_bgr):
    h, w = min(src_bgr.shape[0], dst_bgr.shape[0]), min(src_bgr.shape[1], dst_bgr.shape[1])
    s = cv2.resize(src_bgr, (w, h)); d = cv2.resize(dst_bgr, (w, h))
    mse = ((s.astype(np.float32) - d.astype(np.float32)) ** 2).mean()
    return float(np.clip(1 - mse / (255.0 ** 2), 0, 1))


import urllib.request
def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())


def gen(ref, seed):
    g = build(ref, seed)
    r = submit(g)
    pid = r["prompt_id"]
    print(f"  submitted {ref['id']} pid={pid[:8]} denoise={DENOISE} ipa={IPA_WEIGHT} (v188 params)")
    for _ in range(72):
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
    raw_path = None
    for node_id, info in outputs.items():
        for img in info.get("images", []):
            src = Path(COMFY_INPUT.parent) / "ComfyUI" / "output" / img["filename"]
            if not src.exists():
                src = Path(img.get("abs_path", src))
            if not src.exists():
                continue
            raw_path = str(src)
            break
        if raw_path:
            break
    if not raw_path:
        out_dir = COMFY_INPUT.parent / "ComfyUI" / "output"
        cands = sorted(out_dir.glob(f"v194_{ref['id']}*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        return None

    src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / ref["ref_img"]).convert("RGB")), cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(np.array(Image.open(raw_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
    out_final = JOB / f"v194_{ref['id']}.jpg"
    Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)).save(str(out_final), quality=95)
    hi_before = hist_intersection(src_bgr, dst_bgr)
    hi_after = hist_intersection(src_bgr, matched)
    sd = structural_diff(src_bgr, matched)
    print(f"  saved v194_{ref['id']}.jpg  ({out_final.stat().st_size//1024} KB)")
    print(f"    配色: 前={hi_before:.3f} → 后={hi_after:.3f} | 结构差异={sd:.3f}")
    return str(out_final)


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    for r in REFS:
        if r["id"] not in targets:
            continue
        out = JOB / f"v194_{r['id']}.jpg"
        if out.exists() and out.stat().st_size > 100000:
            print(f"[skip] {r['id']} already done")
            continue
        gen(r, SEED)


if __name__ == "__main__":
    main()
