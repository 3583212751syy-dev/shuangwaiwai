"""
v277 — 复刻 v147/v213 SDXL 整体重绘溶字 + PIL 烧新字

核心：
  1. 用 v213 锁死参数（denoise 0.80 / ipa 0.18 / lora 1.0 / canny 0.25 / tile 0.60）
  2. 关键 trick：正提示要求 ornamental stylized letter-shaped ribbons，
     负提示强烈禁止 readable text / BACARDI / brand name
     → SDXL 把旧字溶解成装饰纹样，而不是重建原字
  3. 后置 Reinhard LAB 配色迁移锁紫
  4. PIL AbrilFatface 矢量烧新字
  5. 输出 3 个变体 + 对照网格
"""
import argparse
import json
import math
import sys
import time
import uuid
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "v277"
JOB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "src"))
import arc_text

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

TARGET_W, TARGET_H = 1024, 1280
INK = (26, 10, 31)

# v147 锁死基线
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  277001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   277101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 277201),
]

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "legible text, readable text, recognizable letters, clear word, English word, brand name, logo word, "
    "spelled out words, specific brand word, recognizable brand, BACARDI, Mooncrest, "
    "calligraphy, scripture, "
    "3d, photographic, painterly, photorealistic bat, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "anatomically incorrect, extra wings, asymmetric error, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "new colors, different color palette, extra colors, color shift, "
    "identical layout to source"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "natural overlap hierarchy, fits the overall composition"
)


def global_pos_for(tag):
    """根据变体微调全局提示中的蝙蝠姿态。"""
    bat_desc = {
        "up":     "an evolved bat silhouette with wings raised upward, talons extended, head tilted",
        "spread": "an evolved bat silhouette with wings spread wide horizontally, powerful stance",
        "fold":   "an evolved bat silhouette with wings folded downward, compact brooding posture",
    }[tag]
    return (
        f"vintage gothic purple spirit brand emblem badge centered on a soft purple background, "
        f"purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F color blocks only, "
        f"a circular emblem with {bat_desc} in the center, "
        f"decorative ornamental stylized letter-shaped ribbons and curlicues woven into the badge design "
        f"as abstract graphical motifs (NOT readable English words, NOT spelled-out words, NOT any brand name), "
        f"the emblem ring is rotated and slightly distorted with baroque flourishes, "
        f"small geometric crescent and triangle accents around the badge ring, "
        f"vintage craft spirits label art style, sharp clean edges, bold emblem"
    )


def build(seed, tag):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ORIG}}
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

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": global_pos_for(tag)}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    # 3 个区域：中心徽章 / 顶部弧带 / 底部饰带
    regions = [
        {
            "x": 0.05, "y": 0.10, "w": 0.90, "h": 0.80, "strength": 1.30,
            "prompt": (
                "a transformed circular purple emblem with an evolved bat silhouette inside, "
                "decorative ornamental stylized letter-shaped ribbons woven into the badge as abstract motifs "
                "(NOT readable words, NOT spelled-out text, NOT any brand name), "
                "baroque flourishes and geometric crescent accents, "
                "keep purple/black palette and craft spirits label style. " + COHESIVE
            )
        },
        {
            "x": 0.10, "y": 0.00, "w": 0.80, "h": 0.15, "strength": 1.10,
            "prompt": (
                "a thin ornamental decorative arched band above the emblem top, "
                "stylized ornamental ribbons and abstract geometric patterns ONLY "
                "(NOT readable text, NOT letter-shaped wordmarks, NOT any word), "
                "purple and black color blocks with sharp edges and baroque flourishes. " + COHESIVE
            )
        },
        {
            "x": 0.20, "y": 0.85, "w": 0.60, "h": 0.13, "strength": 1.05,
            "prompt": (
                "a flat bottom decorative banner with abstract ornamental patterns and stylized "
                "letter-shaped ribbons as decorative motifs (NOT readable English words, NOT spelled-out words), "
                "kept purple and black palette. " + COHESIVE
            )
        },
    ]

    region_nodes = []
    for i, r in enumerate(regions):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"] * REGION_STRENGTH_SCALE}}
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v277_{tag}"}}
    return g


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


def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def poll(pid, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(5)
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=20) as r:
                h = json.loads(r.read())
        except Exception:
            continue
        if pid in h:
            e = h[pid]
            if e.get("status", {}).get("completed"):
                return e
            if e.get("status", {}).get("error"):
                raise RuntimeError(f"ComfyUI error: {e['status']['error']}")
    raise TimeoutError("poll timeout")


def collect_outputs(entry):
    outs = []
    for node in entry.get("outputs", {}).values():
        if "images" in node:
            for im in node["images"]:
                outs.append(COMFY_OUTPUT / im["filename"])
    return outs


def _calibrate_font(text, font_path, start_size, max_w):
    lo, hi = 8, start_size
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        if font.getlength(text) <= max_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def burn_text(img, arc, big, sub, color=INK):
    img = img.convert("RGB")
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)

    fs_arc = int(w * 0.045)
    radius = int(w * 0.46)
    cy = int(h * 0.13) + radius
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=3)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 170:
            break
        fs_arc = int(fs_arc * 0.92)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy), radius=radius,
                                 start_angle_deg=start, end_angle_deg=end,
                                 char_spacing_px=3, flip_180=False)

    fs_big = _calibrate_font(big, FONT, int(h * 0.145), max_w=int(w * 0.82))
    font = ImageFont.truetype(FONT, fs_big)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")

    fs_sub = _calibrate_font(sub, FONT, int(h * 0.075), max_w=int(w * 0.60))
    font = ImageFont.truetype(FONT, fs_sub)
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.028))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="重新生成 ComfyUI 输出")
    args = ap.parse_args()

    orig_pil = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)

    files = []
    for tag, big, sub, arc, seed in SUBJECT_VARIANTS:
        cands = sorted(COMFY_OUTPUT.glob(f"v277_{tag}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands or args.force:
            print(f"[v277] generating {tag} ...")
            g = build(seed, tag)
            r = submit(g)
            pid = r["prompt_id"]
            print(f"  pid={pid[:8]}")
            entry = poll(pid, timeout=900)
            outs = collect_outputs(entry)
            if not outs:
                raise RuntimeError(f"{tag}: no output")
            raw_path = outs[0]
        else:
            raw_path = cands[0]
            print(f"[v277] reuse {tag}: {raw_path.name}")

        # LAB 配色迁移锁紫
        raw = Image.open(raw_path).convert("RGB")
        raw = raw.resize((1552, 2000), Image.LANCZOS)
        raw_bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
        matched_bgr = color_transfer(orig_bgr, raw_bgr, alpha=0.92)
        matched = Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB))
        matched.save(JOB / f"v277_{tag}_matched.png", quality=95)

        # 烧新字
        final = burn_text(matched, arc, big, sub)
        final_path = JOB / f"v277_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"  {tag} final -> {final_path.name}")

    # 对照网格
    orig_small = orig_pil.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v277.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")


if __name__ == "__main__":
    main()
