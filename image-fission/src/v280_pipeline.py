"""
v280 — 强制姿势引导 + 更大重绘区域，彻底拉开蝙蝠裂变

针对 v279 被用户截图指出的问题：三只蝙蝠姿态视觉差异不够大。
解决：用程序把原蝙蝠 mask 形变成 up/spread/fold 三种姿态骨架，
作为 ControlNet Canny 引导图强制注入；重绘区域扩大到 inner badge 圆，
让翅膀有空间上扬/横展/下垂。背景圆环、弧带、文字区仍保留不动。
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
JOB = PROJECT / "jobs" / "v280"
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

cx, cy, r_badge = 776, 746, 300  # inner badge 圆心/半径

# 裂变参数：denoise 拉高 + Canny 姿势引导主导结构 + IPA/TILE 锁风格
DENOISE = 0.95
REFINE_DENOISE = 0.15
IPA_WEIGHT = 0.10
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.65
TILE_STRENGTH = 0.40

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  280001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   280101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 280201),
]

NEG = (
    "text, letters, words, readable text, brand name, BACARDI, logo, watermark, "
    "tarot, mandala, chandelier, filigree, ornamental frame, excessive decoration, "
    "green, brown, gray, beige, blue, color shift, blur, watercolor, photorealistic, "
    "multiple bats, small bat, realistic fur"
)


def _ring_sector_mask(h, w, cx, cy, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) - a0) % 360
    sweep = (a1 - a0) % 360
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    in_angle = ang <= sweep
    return in_ring & in_angle


def _global_bg_color(arr):
    h, w = arr.shape[:2]
    regions = [
        arr[:int(h*0.15), :int(w*0.15)],
        arr[:int(h*0.15), int(w*0.85):],
        arr[int(h*0.85):, :int(w*0.15)],
        arr[int(h*0.85):, int(w*0.85):],
    ]
    samples = np.concatenate([r.reshape(-1, 3) for r in regions], axis=0)
    return np.median(samples, axis=0)


def make_text_free_source():
    """同 v279：全局背景色填充 + 羽化清字，蝙蝠/圆环保留。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]
    ring_outer = 421
    bg = _global_bg_color(arr)

    masks = []
    arc_mask = _ring_sector_mask(h, w, cx, cy, ring_outer - 90, ring_outer + 70, 190, 350)
    masks.append(arc_mask)

    rects = [
        (1000, 1180, 150, 1400),
        (1180, 1350, 350, 1200),
        (720,  880,  300, 620),
        (720,  880,  930, 1250),
    ]
    for y0, y1, x0, x1 in rects:
        m = np.zeros((h, w), dtype=np.uint8)
        m[y0:y1, x0:x1] = 255
        masks.append(m)

    for m in masks:
        m = cv2.dilate(m.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1)
        arr[m > 0] = bg
        m_blur = cv2.GaussianBlur(m.astype(np.float32), (25, 25), 0) / 255.0
        blurred = cv2.GaussianBlur(arr.astype(np.uint8), (25, 25), 0).astype(np.float32)
        for c in range(3):
            arr[:, :, c] = arr[:, :, c] * (1 - m_blur) + blurred[:, :, c] * m_blur

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_bat_mask(size=(1552, 2000)):
    """仅蝙蝠剪影 mask（同 v279）。"""
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
    mask = cv2.dilate(mask, np.ones((12, 12), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return Image.fromarray(mask)


def _transform_silhouette(mask, sx, sy, rot_deg=0.0):
    """以 (cx,cy) 为中心的仿射形变：缩放 + 旋转。"""
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    # 再叠加缩放（在旋转矩阵基础上修改）
    M[0, 0] *= sx
    M[0, 1] *= sx
    M[1, 0] *= sy
    M[1, 1] *= sy
    # 修正中心偏移
    M[0, 2] = cx * (1 - sx)
    M[1, 2] = cy * (1 - sy)
    out = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def make_pose_guides():
    """生成三姿态 Canny 引导边线图（白边黑底）。"""
    bat = np.array(make_bat_mask().convert("L")).astype(np.uint8)
    _, bat = cv2.threshold(bat, 127, 255, cv2.THRESH_BINARY)

    # 形态学获得更干净的剪影
    bat = cv2.morphologyEx(bat, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    bat = cv2.morphologyEx(bat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    poses = {}
    # up：竖向拉高，横向略收（翅膀上扬）
    up = _transform_silhouette(bat, sx=0.70, sy=1.35)
    # spread：横向拉宽，竖向压扁（翅膀平展）
    spread = _transform_silhouette(bat, sx=1.50, sy=0.65)
    # fold： wings down — 宽底窄顶，让模型画出翅膀下垂
    fold = _transform_silhouette(bat, sx=0.95, sy=0.85, rot_deg=0.0)
    # 让 fold 上半部分更窄、下半部分更宽：径向形变（以 cy 为中心，y>cy 时 x 放大）
    h, w = fold.shape
    ys = np.arange(h).reshape(-1, 1).astype(np.float32)
    factor = 1.0 + 0.25 * np.clip((ys - cy) / (h - cy), 0, 1)
    x_map = (np.arange(w).astype(np.float32)[None, :] - cx) / factor + cx
    y_map = np.tile(ys, (1, w))
    fold = cv2.remap(fold, x_map.astype(np.float32), y_map.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    for tag, m in (("up", up), ("spread", spread), ("fold", fold)):
        # 限制在 inner badge 内
        badge = np.zeros_like(m)
        cv2.circle(badge, (cx, cy), r_badge + 20, 255, -1)
        m = cv2.bitwise_and(m, badge)
        # 取边线
        edges = cv2.Canny(m, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        poses[tag] = Image.fromarray(edges).convert("RGB")
        path = COMFY_INPUT / f"v280_pose_{tag}.png"
        poses[tag].save(path, quality=95)
        print(f"[v280] pose guide {tag} -> {path}")
    return poses


def make_inpaint_mask(size=(1552, 2000)):
    """重绘 mask：inner badge 圆，给翅膀足够空间。"""
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r_badge, 255, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return Image.fromarray(mask)


def build(seed, tag, pose_img):
    bat_desc = {
        "up": (
            "bat wings raised steeply upward pointing to top, tall vertical dramatic pose, "
            "elongated silhouette, wings angled 60 degrees up"
        ),
        "spread": (
            "bat wings spread wide horizontally in powerful open stance, broad wingspan, "
            "wings flat and stretched left and right"
        ),
        "fold": (
            "bat wings folded downward draping like dark cloak, wings wrapped around body, "
            "compact hunched posture, wings pointing down"
        ),
    }[tag]

    pos = (
        f"vintage gothic purple spirit brand emblem, circular dark purple badge, "
        f"black bat silhouette in center, {bat_desc}, NO text, NO letters, NO words, "
        f"ornamental geometric flourishes around the outer ring, baroque accents, "
        f"purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F only, "
        f"sharp clean edges, bold emblem, same art style and material as reference"
    )

    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v280_text_free_source.png"}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}

    g["mask_load"] = {"class_type": "LoadImage", "inputs": {"image": "v280_inpaint_mask.png"}}
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

    # Canny 用强制姿势引导图（非原图），直接传入边线图，绕过 preprocessor
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": f"v280_pose_{tag}.png"}}
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v280_{tag}"}}
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
    args = ap.parse_args()

    src_path = COMFY_INPUT / "v280_text_free_source.png"
    mask_path = COMFY_INPUT / "v280_inpaint_mask.png"

    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        print(f"[v280] text-free source -> {src_path}")
    else:
        print(f"[v280] reuse text-free source")

    pose_guides = make_pose_guides()

    if not mask_path.exists() or args.force:
        mask = make_inpaint_mask()
        rgba = Image.merge("RGBA", (mask, mask, mask, mask))
        rgba.save(mask_path)
        print(f"[v280] inpaint mask -> {mask_path}")
    else:
        print(f"[v280] reuse inpaint mask")

    text_free = Image.open(src_path).convert("RGB")
    inpaint_mask = Image.open(mask_path).convert("L")

    files = []
    for tag, big, sub, arc, seed in SUBJECT_VARIANTS:
        cands = sorted(COMFY_OUTPUT.glob(f"v280_{tag}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands or args.force:
            print(f"[v280] generating {tag} ...")
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
            print(f"[v280] reuse {tag}: {raw_path.name}")

        raw = Image.open(raw_path).convert("RGB").resize((1552, 2000), Image.LANCZOS)
        raw_arr = np.array(raw).astype(np.float32)
        bg_arr = np.array(text_free).astype(np.float32)
        mask_arr = np.array(inpaint_mask).astype(np.float32) / 255.0
        mask_arr = np.stack([mask_arr] * 3, axis=-1)
        composed = raw_arr * mask_arr + bg_arr * (1 - mask_arr)
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v280_{tag}_composed.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v280_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"  {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v280.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")


if __name__ == "__main__":
    main()
