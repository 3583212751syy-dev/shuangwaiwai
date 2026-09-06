"""
v278 — 先制无字底图 → SDXL 裂变蝙蝠/徽章 → PIL 烧新字

关键：
  1. 用 PIL/cv2 把原图所有文字区（弧带/主副字/Est/1862）填成周围背景色，
     得到 text_free_source.png。SDXL 的 Canny/IPAdapter 从此看不到任何文字。
  2. v213 锁死参数整体重绘：denoise 0.80 / ipa 0.18 / canny 0.25 / tile 0.60
     正提示强约束"no text, no letters, no words"，只让蝙蝠/徽章裂变。
  3. LAB 配色迁移锁紫底。
  4. PIL AbrilFatface 矢量烧新字。
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
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "v278"
JOB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "src"))
import arc_text

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"

TARGET_W, TARGET_H = 1024, 1280
INK = (26, 10, 31)

DENOISE = 0.85
IPA_WEIGHT = 0.15
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.20
TILE_STRENGTH = 0.55

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  278001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   278101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 278201),
]

NEG = (
    "text, letters, words, readable text, brand name, BACARDI, logo, watermark, "
    "tarot, mandala, chandelier, filigree, ornamental frame, excessive decoration, "
    "green, brown, gray, beige, blue, color shift, blur, watercolor, photorealistic"
)


def _ring_sector_mask(h, w, cx, cy, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) - a0) % 360
    sweep = (a1 - a0) % 360
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    in_angle = ang <= sweep
    return in_ring & in_angle


def _bg_color(arr, mask=None, percentile=75):
    region = arr[mask] if mask is not None else arr.reshape(-1, 3)
    if region.size == 0:
        return arr.mean(axis=(0, 1))
    return np.percentile(region, percentile, axis=0)


def make_text_free_source():
    """生成无字原图：弧带/主副字/Est 全部填平，保留蝙蝠+圆环+背景。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]
    cx, cy = 776, 746
    ring_outer = 421

    # 顶部弧带：覆盖整个上半圆环；用全局背景色（而非弧带本身颜色）填充，
    # 再用强羽化消融硬边，避免留下浅色"盖子"
    arc_mask = _ring_sector_mask(h, w, cx, cy, ring_outer - 120, ring_outer + 200, 190, 350)
    light_bg = _bg_color(arr, None, percentile=70)  # 全局浅紫背景
    arr[arc_mask] = light_bg
    # 对弧带边界做强羽化
    arc_u8 = (arc_mask.astype(np.uint8)) * 255
    arc_u8 = cv2.GaussianBlur(arc_u8, (31, 31), 0)
    arc_alpha = arc_u8.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(arr.astype(np.uint8), (31, 31), 0)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - arc_alpha) + blurred[:, :, c] * arc_alpha

    # 主/副/Est 矩形：用更宽全局浅紫背景色填充
    light_bg = _bg_color(arr, None, percentile=70)
    rects = [
        (975, 1175, 155, 1395),   # 主字
        (1175, 1345, 355, 1200),  # 副字
        (700, 900, 325, 585),     # Est.
        (700, 900, 975, 1255),    # 1862
    ]
    for y0, y1, x0, x1 in rects:
        arr[y0:y1, x0:x1] = light_bg
        arr[y0:y1, x0:x1] = cv2.GaussianBlur(arr[y0:y1, x0:x1].astype(np.uint8), (9, 9), 0)

    # 蝙蝠区不做任何处理——保留原蝙蝠作为 SDXL 结构锚点
    return Image.fromarray(arr.astype(np.uint8))


def build(seed, tag):
    bat_desc = {
        "up": "wings raised upward, talons extended, head tilted up",
        "spread": "wings spread wide horizontally, powerful stance",
        "fold": "wings folded downward, compact brooding posture",
    }[tag]

    pos = (
        f"vintage gothic purple spirit brand emblem centered on soft purple background, "
        f"circular dark purple badge with black bat silhouette, {bat_desc}, "
        f"NO text, NO letters, NO words, NO brand name, NO readable glyphs, "
        f"ornamental geometric flourishes around the ring, baroque accents, "
        f"purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F only, "
        f"vintage craft spirits label art, sharp clean edges, bold emblem"
    )

    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v278_text_free_source.png"}}
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

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": pos}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG}}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["24", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 28, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["24", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.12}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v278_{tag}"}}
    return g


def color_transfer(src_bgr, dst_bgr, alpha=0.92):
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
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_path = COMFY_INPUT / "v278_text_free_source.png"
    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        print(f"[v278] text-free source -> {src_path}")
    else:
        print(f"[v278] reuse text-free source")

    orig_pil = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)

    files = []
    for tag, big, sub, arc, seed in SUBJECT_VARIANTS:
        cands = sorted(COMFY_OUTPUT.glob(f"v278_{tag}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands or args.force:
            print(f"[v278] generating {tag} ...")
            g = build(seed, tag)
            r = submit(g)
            pid = r["prompt_id"]
            print(f"  pid={pid[:8]}")
            entry = poll(pid, timeout=900)
            outs = []
            for n in entry.get("outputs", {}).values():
                if "images" in n:
                    outs += [COMFY_OUTPUT / im["filename"] for im in n["images"]]
            if not outs:
                raise RuntimeError(f"{tag}: no output")
            raw_path = outs[0]
        else:
            raw_path = cands[0]
            print(f"[v278] reuse {tag}: {raw_path.name}")

        raw = Image.open(raw_path).convert("RGB")
        raw = raw.resize((1552, 2000), Image.LANCZOS)
        raw_bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
        matched_bgr = color_transfer(orig_bgr, raw_bgr, alpha=0.92)
        matched = Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB))
        matched.save(JOB / f"v278_{tag}_matched.png", quality=95)

        final = burn_text(matched, arc, big, sub)
        final_path = JOB / f"v278_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"  {tag} final -> {final_path.name}")

    orig_small = orig_pil.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v278.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")


if __name__ == "__main__":
    main()
