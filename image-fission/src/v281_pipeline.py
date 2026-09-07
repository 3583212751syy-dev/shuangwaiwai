"""
v281 — 严格只重绘蝙蝠剪影，内徽章背景 100% 保留；文字区局部采样填充；姿势骨架更极端。

修复 v280 肉眼可见的问题：
  1. 内徽章背景颜色被改 → 重绘/合成 mask 都缩回蝙蝠剪影本身，不动紫色 badge。
  2. 文字区有遮挡/残影 → 改从文字区紧邻背景采样局部颜色 + 匹配噪点 + 大羽化，不用扁平全局色。
  3. 蝙蝠裂变仍不够 → 骨架更极端（up 拉高 45%/spread 拉宽 70%/fold 上窄下宽），Canny 强度提到 0.85。

流程：
  1. text_free_source：局部背景填充清字，保留蝙蝠+圆环，输出 _source_check 供自检。
  2. bat_mask：仅蝙蝠剪影，边缘轻羽化（合成用）。
  3. inpaint_mask：蝙蝠剪影外扩 40px（给 ComfyUI 上下文，但最终合成只取 bat_mask）。
  4. pose guides：程序生成三种强制骨架，作为 Canny 边线图。
  5. ComfyUI：text_free_source + inpaint_mask → VAEEncodeForInpaint；Canny 姿势引导；Tile 锁色。
  6. 合成：SDXL 输出按 bat_mask 拼回 text_free_source → 背景绝对不变。输出 _compose_check 供自检。
  7. PIL 烧新字。
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
JOB = PROJECT / "jobs" / "v281"
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

# 蝙蝠中心/inner badge 半径（仅用于限制 mask 搜索范围）
cx, cy, r_badge = 776, 746, 300

# 裂变参数：Canny 主导姿势，IPA/TILE 锁材质，denoise 适中避免糊化
DENOISE = 0.92
REFINE_DENOISE = 0.12
IPA_WEIGHT = 0.12
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.85
TILE_STRENGTH = 0.45

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  281001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   281101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 281201),
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


def _global_bg_color(arr):
    """从四角安全区采样全局背景色。"""
    h, w = arr.shape[:2]
    regions = [
        arr[:int(h*0.18), :int(w*0.18)],
        arr[:int(h*0.18), int(w*0.82):],
        arr[int(h*0.82):, :int(w*0.18)],
        arr[int(h*0.82):, int(w*0.82):],
    ]
    samples = np.concatenate([r.reshape(-1, 3) for r in regions], axis=0)
    return np.median(samples, axis=0)


def _sample_local_bg(arr, mask, dilate_px=18):
    """从 mask 外围环采样局部背景颜色与方差。"""
    d = cv2.dilate(mask.astype(np.uint8), np.ones((dilate_px, dilate_px), np.uint8), iterations=1)
    e = cv2.erode(mask.astype(np.uint8), np.ones((max(3, dilate_px//6), max(3, dilate_px//6)), np.uint8), iterations=1)
    ring = (d > 0) & (e == 0)
    if ring.sum() < 100:
        return _global_bg_color(arr), np.array([3.0, 3.0, 3.0])
    pixels = arr[ring].reshape(-1, 3)
    return np.median(pixels, axis=0), pixels.std(axis=0)


def _texture_fill(arr, mask, feather=35):
    """局部背景色填充 + 匹配方差的高斯噪点 + 羽化。"""
    h, w = arr.shape[:2]
    bg, std = _sample_local_bg(arr, mask)
    filled = arr.copy()
    m_bin = (mask > 127).astype(np.uint8)
    filled[m_bin > 0] = bg
    # 加噪点匹配背景纹理
    noise = np.random.normal(0, std, size=(h, w, 3)).astype(np.float32)
    # 只在 mask 内加噪，避免污染周围
    filled = filled + noise * (m_bin[:, :, None].astype(np.float32))
    filled = np.clip(filled, 0, 255)
    # 羽化混合
    m_blur = cv2.GaussianBlur(mask.astype(np.float32), (feather*2+1, feather*2+1), 0) / 255.0
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - m_blur) + filled[:, :, c] * m_blur
    return arr


def _clean_small_text(arr, mask, radius=3):
    """对小块文字（Est/1862/小划痕）用 cv2.inpaint 局部修补。"""
    m = (mask > 127).astype(np.uint8) * 255
    inp = cv2.inpaint(arr.astype(np.uint8), m, radius, cv2.INPAINT_NS)
    m_blur = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0) / 255.0
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - m_blur) + inp[:, :, c] * m_blur
    return arr


def make_text_free_source():
    """清字底图：局部采样填充 + 小字 inpaint，保留蝙蝠+圆环。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]
    ring_outer = 421

    masks = []

    # 1. 顶部弧带（覆盖 LA CASA DEL MURCIELAGO）
    arc_mask = _ring_sector_mask(h, w, cx, cy, ring_outer - 100, ring_outer + 90, 185, 355)
    masks.append(("arc", arc_mask))

    # 2. 主/副/Est/1862 矩形区
    rects = [
        ("main",   1000, 1185, 130, 1420),  # BACARDÍ
        ("sub",    1185, 1365, 320, 1230),  # MCHEART
        ("est",     720,  900,  280,  640),  # Est.
        ("year",    720,  900,  910, 1270),  # 1862
        ("tri",    1470, 1540, 680,  875),  # 底部小三角周围文字残迹
    ]
    for name, y0, y1, x0, x1 in rects:
        m = np.zeros((h, w), dtype=np.uint8)
        m[y0:y1, x0:x1] = 255
        masks.append((name, m))

    # 先分别局部填充大区域
    for name, m in masks:
        arr = _texture_fill(arr, m, feather=45 if name == "arc" else 35)

    # 再对小块文字和装饰划痕做一次小范围 inpaint（只覆盖 Est/1862/弧带内残留）
    small_mask = np.zeros((h, w), dtype=np.uint8)
    small_mask[720:900, 280:640] = 255
    small_mask[720:900, 910:1270] = 255
    # 弧带内部可能有残留装饰字，做一次整体轻 inpaint
    arc_inpaint = _ring_sector_mask(h, w, cx, cy, ring_outer - 80, ring_outer + 70, 190, 350)
    small_mask = cv2.bitwise_or(small_mask, arc_inpaint.astype(np.uint8) * 255)
    arr = _clean_small_text(arr, small_mask, radius=3)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_bat_mask(size=(1552, 2000)):
    """精确蝙蝠剪影 mask，限定 inner badge 圆内，边缘轻羽化。"""
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
    """以 (cx,cy) 为中心的仿射形变 + 旋转。"""
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    M[0, 0] *= sx
    M[0, 1] *= sx
    M[1, 0] *= sy
    M[1, 1] *= sy
    M[0, 2] = cx * (1 - sx) - cy * sx * math.sin(math.radians(-rot_deg)) + cx * math.cos(math.radians(rot_deg)) * sx
    M[1, 2] = cy * (1 - sy) + cx * sy * math.sin(math.radians(-rot_deg)) + cy * math.cos(math.radians(rot_deg)) * sy
    # 更简单的中心修正
    M[0, 2] = cx - sx * (cx * math.cos(math.radians(rot_deg)) - cy * math.sin(math.radians(rot_deg)))
    M[1, 2] = cy - sy * (cx * math.sin(math.radians(rot_deg)) + cy * math.cos(math.radians(rot_deg)))
    out = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def make_pose_guides():
    """生成三姿态 Canny 引导边线图（白边黑底），骨架更极端。"""
    bat = np.array(make_bat_mask().convert("L")).astype(np.uint8)
    _, bat = cv2.threshold(bat, 127, 255, cv2.THRESH_BINARY)

    # 取更粗的边线，强迫 SDXL 跟骨架
    bat = cv2.morphologyEx(bat, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    bat = cv2.morphologyEx(bat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    poses = {}
    # up：竖向拉高 45%，横向收紧 35%，略后仰
    up = _transform_silhouette(bat, sx=0.65, sy=1.45, rot_deg=-3)
    # spread：横向拉宽 70%，竖向压扁 40%，翅膀平张
    spread = _transform_silhouette(bat, sx=1.70, sy=0.60, rot_deg=0)
    # fold：上窄下宽，顶部压缩，底部外扩，翅膀下垂
    fold_base = _transform_silhouette(bat, sx=1.05, sy=0.90, rot_deg=2)
    h, w = fold_base.shape
    ys = np.arange(h).reshape(-1, 1).astype(np.float32)
    # y>cy 时横向放大（翅膀下垂变宽），y<cy 时横向缩小
    factor = 1.0 + 0.30 * np.clip((ys - cy) / (h - cy), 0, 1) - 0.10 * np.clip((cy - ys) / cy, 0, 1)
    x_map = (np.arange(w).astype(np.float32)[None, :] - cx) / factor + cx
    y_map = np.tile(ys, (1, w))
    fold = cv2.remap(fold_base, x_map.astype(np.float32), y_map.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    for tag, m in (("up", up), ("spread", spread), ("fold", fold)):
        # 限制在 inner badge 内
        badge = np.zeros_like(m)
        cv2.circle(badge, (cx, cy), r_badge, 255, -1)
        m = cv2.bitwise_and(m, badge)
        edges = cv2.Canny(m, 50, 150)
        edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)
        poses[tag] = Image.fromarray(edges).convert("RGB")
        path = COMFY_INPUT / f"v281_pose_{tag}.png"
        poses[tag].save(path, quality=95)
        print(f"[v281] pose guide {tag} -> {path}")
    return poses


def make_inpaint_mask(size=(1552, 2000)):
    """ComfyUI 重绘 mask：蝙蝠剪影外扩 40px，给 SDXL 上下文。"""
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
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v281_text_free_source.png"}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}

    g["mask_load"] = {"class_type": "LoadImage", "inputs": {"image": "v281_inpaint_mask.png"}}
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

    g["20"] = {"class_type": "LoadImage", "inputs": {"image": f"v281_pose_{tag}.png"}}
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v281_{tag}"}}
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

    src_path = COMFY_INPUT / "v281_text_free_source.png"
    mask_path = COMFY_INPUT / "v281_inpaint_mask.png"
    bat_mask_path = COMFY_INPUT / "v281_bat_mask.png"

    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        # 自检图：左原图 / 右清字底图
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v281] text-free source -> {src_path}")
        print(f"[v281] source check -> {JOB / '_source_check.png'}")
    else:
        print(f"[v281] reuse text-free source")

    # 生成并保存 bat_mask 用于最终合成
    if not bat_mask_path.exists() or args.force:
        bm = make_bat_mask()
        bm.save(bat_mask_path)
        print(f"[v281] bat mask -> {bat_mask_path}")
    else:
        print(f"[v281] reuse bat mask")

    # 生成 pose guides
    pose_guides = make_pose_guides()

    if not mask_path.exists() or args.force:
        mask = make_inpaint_mask()
        rgba = Image.merge("RGBA", (mask, mask, mask, mask))
        rgba.save(mask_path)
        print(f"[v281] inpaint mask -> {mask_path}")
    else:
        print(f"[v281] reuse inpaint mask")

    text_free = Image.open(src_path).convert("RGB")
    bat_mask = Image.open(bat_mask_path).convert("L")
    bat_mask_arr = np.array(bat_mask).astype(np.float32) / 255.0
    bat_mask_arr = np.stack([bat_mask_arr] * 3, axis=-1)

    files = []
    for tag, big, sub, arc, seed in SUBJECT_VARIANTS:
        cands = sorted(COMFY_OUTPUT.glob(f"v281_{tag}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands or args.force:
            print(f"[v281] generating {tag} ...")
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
            print(f"[v281] reuse {tag}: {raw_path.name}")

        raw = Image.open(raw_path).convert("RGB").resize((1552, 2000), Image.LANCZOS)
        raw_arr = np.array(raw).astype(np.float32)
        bg_arr = np.array(text_free).astype(np.float32)
        # 合成：只在 bat_mask 区域取 SDXL 输出，其余 100% 原图
        composed = raw_arr * bat_mask_arr + bg_arr * (1 - bat_mask_arr)
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v281_{tag}_composed.png", quality=95)

        # 自检图：左清字底图 / 右合成后（未烧字）
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v281_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"  {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v281.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")


if __name__ == "__main__":
    main()
