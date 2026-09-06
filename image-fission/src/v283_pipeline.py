"""
v283 — 精确检测并填充黑色文字像素，保留圆环/横幅/卷草；只重绘蝙蝠剪影。

修复 v282 问题：
  1. v282 用大块 preserve mask 把弧带/主字区也保护了，导致旧字残留。
  2. v282 大模糊破坏了圆环清晰度。

v283 改法：
  1. 文字区按区域检测"比局部背景暗很多"的像素（即黑色文字），只填这些像素。
  2. 用局部背景色 + 匹配噪点 + 羽化修补，横幅/圆环/卷草保持原样。
  3. 圆环、内徽章、蝙蝠 100% 不动；仅蝙蝠区交给 SDXL 重绘。
  4. 合成用蝙蝠剪影 mask，确保背景色不变。
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
JOB = PROJECT / "jobs" / "v283"
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

cx, cy, r_badge = 776, 746, 300
ring_outer = 421

DENOISE = 0.92
REFINE_DENOISE = 0.12
IPA_WEIGHT = 0.12
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.85
TILE_STRENGTH = 0.45

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  283001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   283101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 283201),
]

NEG = (
    "text, letters, words, readable text, brand name, BACARDI, logo, watermark, "
    "tarot, mandala, chandelier, filigree, ornamental frame, excessive decoration, "
    "green, brown, gray, beige, blue, color shift, blur, watercolor, photorealistic, "
    "multiple bats, small bat, realistic fur, white bat, red bat, orange bat"
)


def _ring_sector_mask(h, w, cx, cy, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) - a0) % 360
    sweep = (a1 - a0) % 360
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    in_angle = ang <= sweep
    return in_ring & in_angle


def _detect_text_pixels(arr, region_mask, threshold=35):
    """在 region_mask 内检测比局部背景明显暗的像素（文字）并返回二值 mask。"""
    h, w = arr.shape[:2]
    region = region_mask.astype(np.uint8) * 255
    # 从 region 外围 12px 采样局部背景
    dilated = cv2.dilate(region, np.ones((12, 12), np.uint8), iterations=1)
    eroded = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=1)
    ring = (dilated > 0) & (eroded == 0)
    if ring.sum() < 50:
        # 退回到全局
        bg = np.median(arr.reshape(-1, 3), axis=0)
    else:
        bg = np.median(arr[ring].reshape(-1, 3), axis=0)
    maxc = np.max(arr, axis=2)
    bg_max = np.max(bg)
    dark = (maxc < bg_max - threshold) & (region_mask > 0)
    # 形态学清理
    dark = dark.astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return dark


def _fill_text_pixels(arr, text_mask, feather=15, noise_scale=0.35):
    """用局部背景色 + 轻噪点填充文字像素，羽化边缘，并做轻模糊让填充区与背景融合。"""
    h, w = arr.shape[:2]
    text_mask_u8 = (text_mask > 127).astype(np.uint8) * 255
    dilated = cv2.dilate(text_mask_u8, np.ones((10, 10), np.uint8), iterations=1)
    eroded = cv2.erode(text_mask_u8, np.ones((2, 2), np.uint8), iterations=1)
    ring = (dilated > 0) & (eroded == 0)
    if ring.sum() < 50:
        bg = np.median(arr.reshape(-1, 3), axis=0)
        std = np.array([3.0, 3.0, 3.0])
    else:
        pixels = arr[ring].reshape(-1, 3)
        bg = np.median(pixels, axis=0)
        # 限制方差，避免噪点过重
        std = np.clip(pixels.std(axis=0), 1.5, 5.0)

    filled = arr.copy()
    filled[text_mask_u8 > 0] = bg

    # 给填充区加轻噪点并轻模糊，避免颗粒感差异
    noise = np.random.normal(0, std * noise_scale, size=(h, w, 3)).astype(np.float32)
    filled = filled + noise * (text_mask_u8[:, :, None] > 0)
    filled = cv2.GaussianBlur(filled.astype(np.uint8), (5, 5), 0).astype(np.float32)
    filled = np.clip(filled, 0, 255)

    m_blur = cv2.GaussianBlur(text_mask_u8.astype(np.float32), (feather*2+1, feather*2+1), 0) / 255.0
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - m_blur) + filled[:, :, c] * m_blur
    return arr


def make_text_free_source():
    """精确去字：弧带只填文字像素保留横幅；主/副/Est/1862 直接填矩形区；
    最后对圆环外背景做统一大模糊 + 加噪，消除所有填充边界。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]

    # 1. 弧带文字：低阈值检测 + 强膨胀覆盖字边缘/残留
    arc_region = _ring_sector_mask(h, w, cx, cy, ring_outer - 95, ring_outer + 65, 185, 355)
    inner_exclude = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 30, 0, 360)
    arc_region = arc_region & (~inner_exclude)
    arc_text_mask = _detect_text_pixels(arr, arc_region, threshold=12)
    arc_text_mask = cv2.dilate(arc_text_mask, np.ones((7, 7), np.uint8), iterations=3)
    arc_text_mask = cv2.morphologyEx(arc_text_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    arr = _fill_text_pixels(arr, arc_text_mask, feather=21)

    # 2. 主/副/Est./1862/三角：这些区域在圆环外且无装饰，直接矩形填充
    rects = [
        (1000, 1185, 80, 1470, 25),   # BACARDÍ 主字
        (1185, 1380, 240, 1310, 25),  # MCHEART 副字
        (700,  910,  240,  680, 25),  # Est.
        (700,  910,  870, 1310, 25),  # 1862
        (1450, 1550, 620,  930, 25),  # 底部三角
    ]
    for y0, y1, x0, x1, feather in rects:
        region = np.zeros((h, w), dtype=np.uint8)
        region[y0:y1, x0:x1] = 255
        region = cv2.dilate(region, np.ones((7, 7), np.uint8), iterations=1)
        arr = _fill_text_pixels(arr, region, feather=feather)

    # 3. 对圆环外背景做大模糊 + 均匀噪点，彻底消除填充边界
    # 保护 mask：内徽章 + 圆环 + 一点卷草
    preserve = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(preserve, (cx, cy), ring_outer + 12, 255, -1)
    # 排除 Est/1862 区域（需要被模糊）
    for y0, y1, x0, x1, _ in [(700, 910, 240, 680, 25), (700, 910, 870, 1310, 25)]:
        preserve[y0:y1, x0:x1] = 0
    preserve = cv2.GaussianBlur(preserve.astype(np.float32), (21, 21), 0) / 255.0

    blurred = cv2.GaussianBlur(arr.astype(np.uint8), (21, 21), 0).astype(np.float32)
    # 全局轻噪点
    bg_std = np.clip(np.std(arr.reshape(-1, 3), axis=0), 1.5, 4.0)
    noise = np.random.normal(0, bg_std * 0.4, size=(h, w, 3)).astype(np.float32)
    blurred = np.clip(blurred + noise, 0, 255)

    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * preserve + blurred[:, :, c] * (1 - preserve)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_bat_mask(size=(1552, 2000)):
    w, h = size
    src = np.array(make_text_free_source()).astype(np.float32)
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge, 255, -1)

    R, G, B = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(src, axis=2)
    dark_neutral = (maxc < 80) & (stdc < 25)

    mask = dark_neutral.astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, badge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return Image.fromarray(mask)


def _transform_silhouette(mask, sx, sy, rot_deg=0.0):
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    M[0, 0] *= sx
    M[0, 1] *= sx
    M[1, 0] *= sy
    M[1, 1] *= sy
    M[0, 2] = cx - sx * (cx * math.cos(math.radians(rot_deg)) - cy * math.sin(math.radians(rot_deg)))
    M[1, 2] = cy - sy * (cx * math.sin(math.radians(rot_deg)) + cy * math.cos(math.radians(rot_deg)))
    out = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def make_pose_guides():
    bat = np.array(make_bat_mask().convert("L")).astype(np.uint8)
    _, bat = cv2.threshold(bat, 127, 255, cv2.THRESH_BINARY)

    bat = cv2.morphologyEx(bat, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    bat = cv2.morphologyEx(bat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    poses = {}
    up = _transform_silhouette(bat, sx=0.65, sy=1.45, rot_deg=-3)
    spread = _transform_silhouette(bat, sx=1.70, sy=0.60, rot_deg=0)
    fold_base = _transform_silhouette(bat, sx=1.05, sy=0.90, rot_deg=2)
    h, w = fold_base.shape
    ys = np.arange(h).reshape(-1, 1).astype(np.float32)
    factor = 1.0 + 0.30 * np.clip((ys - cy) / (h - cy), 0, 1) - 0.10 * np.clip((cy - ys) / cy, 0, 1)
    x_map = (np.arange(w).astype(np.float32)[None, :] - cx) / factor + cx
    y_map = np.tile(ys, (1, w))
    fold = cv2.remap(fold_base, x_map.astype(np.float32), y_map.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    for tag, m in (("up", up), ("spread", spread), ("fold", fold)):
        badge = np.zeros_like(m)
        cv2.circle(badge, (cx, cy), r_badge, 255, -1)
        m = cv2.bitwise_and(m, badge)
        edges = cv2.Canny(m, 50, 150)
        edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)
        poses[tag] = Image.fromarray(edges).convert("RGB")
        path = COMFY_INPUT / f"v283_pose_{tag}.png"
        poses[tag].save(path, quality=95)
        print(f"[v283] pose guide {tag} -> {path}")
    return poses


def make_inpaint_mask(size=(1552, 2000)):
    w, h = size
    bat = np.array(make_bat_mask().convert("L")).astype(np.uint8)
    _, bat = cv2.threshold(bat, 80, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(bat, np.ones((40, 40), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return Image.fromarray(mask)


def build(seed, tag, pose_img):
    bat_desc = {
        "up": (
            "bat wings raised steeply upward, tall vertical dramatic pose, "
            "elongated silhouette, wings pointing to top center"
        ),
        "spread": (
            "bat wings spread wide horizontally, powerful broad wingspan, "
            "wings flat and stretched left and right, open stance"
        ),
        "fold": (
            "bat wings folded downward draping like dark cloak, "
            "wings wrapped around lower body, compact hunched posture"
        ),
    }[tag]

    pos = (
        f"black bat silhouette only, centered in dark purple circular badge, {bat_desc}, "
        f"NO text, NO letters, NO words, NO background change, "
        f"gothic emblem style, sharp clean edges, deep violet and black only"
    )

    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v283_text_free_source.png"}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}

    g["mask_load"] = {"class_type": "LoadImage", "inputs": {"image": "v283_inpaint_mask.png"}}
    g["mask"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["mask_load", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}
    g["mask_conv"] = {"class_type": "ImageToMask", "inputs": {"image": ["mask", 0], "channel": "red"}}

    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["3", 0], "vae": ["1", 2], "mask": ["mask_conv", 0], "grow_mask_by": 8}}

    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.80,
        "noise": 0.08, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}

    g["20"] = {"class_type": "LoadImage", "inputs": {"image": f"v283_pose_{tag}.png"}}
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
        "latent_image": ["4", 0], "seed": seed, "steps": 32, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["24", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": REFINE_DENOISE}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v283_{tag}"}}
    return g


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
    cy_arc = int(h * 0.13) + radius
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=3)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 170:
            break
        fs_arc = int(fs_arc * 0.92)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy_arc), radius=radius,
                                 start_angle_deg=start, end_angle_deg=end,
                                 char_spacing_px=3, flip_180=False)

    fs_big = _calibrate_font(big, FONT, int(h * 0.145), max_w=int(w * 0.82))
    font = ImageFont.truetype(FONT, fs_big)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")

    fs_sub = _calibrate_font(sub, FONT, int(h * 0.075), max_w=int(w * 0.60))
    font = ImageFont.truetype(FONT, fs_sub)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.028))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--source-only", action="store_true", help="只生成 text_free_source 并出 check 图")
    args = ap.parse_args()

    src_path = COMFY_INPUT / "v283_text_free_source.png"
    mask_path = COMFY_INPUT / "v283_inpaint_mask.png"
    bat_mask_path = COMFY_INPUT / "v283_bat_mask.png"

    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v283] text-free source -> {src_path}")
        print(f"[v283] source check -> {JOB / '_source_check.png'}")
    else:
        print(f"[v283] reuse text-free source")

    if args.source_only:
        print("[v283] source-only mode, exit.")
        return

    if not bat_mask_path.exists() or args.force:
        bm = make_bat_mask()
        bm.save(bat_mask_path)
        print(f"[v283] bat mask -> {bat_mask_path}")
    else:
        print(f"[v283] reuse bat mask")

    pose_guides = make_pose_guides()

    if not mask_path.exists() or args.force:
        mask = make_inpaint_mask()
        rgba = Image.merge("RGBA", (mask, mask, mask, mask))
        rgba.save(mask_path)
        print(f"[v283] inpaint mask -> {mask_path}")
    else:
        print(f"[v283] reuse inpaint mask")

    text_free = Image.open(src_path).convert("RGB")
    bat_mask = Image.open(bat_mask_path).convert("L")
    bat_mask_arr = np.array(bat_mask).astype(np.float32) / 255.0
    bat_mask_arr = np.stack([bat_mask_arr] * 3, axis=-1)

    files = []
    for tag, big, sub, arc, seed in SUBJECT_VARIANTS:
        cands = sorted(COMFY_OUTPUT.glob(f"v283_{tag}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands or args.force:
            print(f"[v283] generating {tag} ...")
            g = build(seed, tag, pose_guides[tag])
            r = submit(g)
            pid = r["prompt_id"]
            print(f"  pid={pid[:8]}")
            entry = poll(pid, timeout=1800)
            outs = []
            for n in entry.get("outputs", {}).values():
                if "images" in n:
                    outs += [COMFY_OUTPUT / im["filename"] for im in n["images"]]
            if not outs:
                raise RuntimeError(f"{tag}: no output")
            raw_path = outs[0]
        else:
            raw_path = cands[0]
            print(f"[v283] reuse {tag}: {raw_path.name}")

        raw = Image.open(raw_path).convert("RGB").resize((1552, 2000), Image.LANCZOS)
        raw_arr = np.array(raw).astype(np.float32)
        bg_arr = np.array(text_free).astype(np.float32)
        composed = raw_arr * bat_mask_arr + bg_arr * (1 - bat_mask_arr)
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v283_{tag}_composed.png", quality=95)

        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v283_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"  {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v283.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")


if __name__ == "__main__":
    main()
