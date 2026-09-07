"""
v276 — PIL 精确裂变（本地 ComfyUI 零 API）

目标：
  1. 旧字彻底清干净（不调用 SDXL 重绘文字区）
  2. 蝙蝠主体裂变（镜像 / 缩放）
  3. 新字清晰不糊（AbrilFatface 矢量渲染）

核心修正：
  - 用 ring-sector 精确覆盖顶部 ribbon，整区填平旧字
  - 主/副/Est 用全局浅紫背景色整区填平
  - 精确提取蝙蝠剪影（非整个 badge 暗区），小半径 inpaint 擦除
  - 裂变只用 mirror + scale，避免旋转产生的方块/clip 伪影
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "src"
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "v276"
JOB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))
import arc_text

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

TARGET_W, TARGET_H = 1024, 1280
ORIG_W, ORIG_H = 1552, 2000

CENTER = (776, 746)
RING_OUTER = 421
INK = (26, 10, 31)

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  "mirror", 1.06),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   "mirror", 1.00),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", "mirror", 0.94),
]


def _ring_sector_mask(h, w, cx, cy, r_in, r_out, a0, a1):
    """环形扇区 bool mask；角度 0=3点钟，逆时针为正。"""
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) - a0) % 360
    sweep = (a1 - a0) % 360
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    in_angle = ang <= sweep
    return in_ring & in_angle


def _bg_color(arr, mask=None, exclude_dark_thr=95, percentile=85):
    """采样背景色：mask 内排除暗像素后取高分位。"""
    region = arr[mask] if mask is not None else arr.reshape(-1, 3)
    if region.size == 0:
        return arr.mean(axis=(0, 1))
    gray = region.mean(axis=1)
    bright = region[gray > exclude_dark_thr]
    if len(bright) < 50:
        bright = region
    return np.percentile(bright, percentile, axis=0)


def extract_bat_silhouette(arr):
    """精确提取蝙蝠剪影：badge 内部近黑连通形状。"""
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - CENTER[0]) ** 2 + (ys - CENTER[1]) ** 2
    inside_badge = d2 < (RING_OUTER - 8) ** 2
    bat = (gray < 35) & inside_badge  # 只取最黑的蝙蝠
    bat = cv2.morphologyEx(bat.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    bat = cv2.morphologyEx(bat, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return bat.astype(bool)


def make_clean_base():
    """生成干净底图：先擦蝙蝠，再填旧字，保留背景与圆环。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]

    # 1) 先擦原蝙蝠：在原始图上做，周围徽章色自然匹配
    bat_mask = extract_bat_silhouette(arr)
    bm = (bat_mask.astype(np.uint8)) * 255
    arr = cv2.inpaint(arr.astype(np.uint8), bm, 3, cv2.INPAINT_NS).astype(np.float32)

    # 2) 顶部弧带：ring-sector 精确覆盖 ribbon，整区硬填 ribbon 背景色
    arc_mask = _ring_sector_mask(h, w, CENTER[0], CENTER[1],
                                 RING_OUTER + 2, RING_OUTER + 130, 155, 385)
    # 用更亮的 ribbon 背景色，并把 arc 区内所有偏暗像素（含旧字 ghost）全部覆盖
    ribbon_bg = _bg_color(arr, arc_mask, exclude_dark_thr=150, percentile=95)
    arc_dark = arc_mask & (arr.mean(axis=2) < 160)
    arr[arc_dark] = ribbon_bg
    arr[arc_mask] = ribbon_bg  # 二次覆盖确保无残留
    blurred = cv2.GaussianBlur(arr.astype(np.uint8), (7, 7), 0)
    arr = (arr * 0.80 + blurred * 0.20)

    # 3) 主/副/Est：用全局浅紫背景色整区填平
    light_bg = _bg_color(arr, None, exclude_dark_thr=110, percentile=75)
    for y0, y1, x0, x1 in [(975, 1175, 155, 1395), (1175, 1345, 355, 1200),
                           (700, 900, 325, 585), (700, 900, 975, 1255)]:
        arr[y0:y1, x0:x1] = light_bg
        arr[y0:y1, x0:x1] = cv2.GaussianBlur(arr[y0:y1, x0:x1].astype(np.uint8), (7, 7), 0)

    return Image.fromarray(arr.astype(np.uint8))


def extract_bat_patch():
    """从原图提取带 alpha 的蝙蝠 patch。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig)
    bat_mask = extract_bat_silhouette(arr)

    # tight bbox of bat
    ys, xs = np.where(bat_mask)
    if len(xs) == 0:
        x, y, bw, bh = 522, 518, 508, 430
    else:
        x, x2 = xs.min(), xs.max()
        y, y2 = ys.min(), ys.max()
        bw, bh = x2 - x + 1, y2 - y + 1

    crop = orig.crop((x, y, x + bw, y + bh)).convert("RGBA")
    pa = np.array(crop)
    pa[..., 3] = (bat_mask[y:y + bh, x:x + bw].astype(np.uint8)) * 255
    return Image.fromarray(pa), (x + bw // 2, y + bh // 2)


def transform_bat(bat, mode, scale):
    if mode == "mirror":
        bat = bat.transpose(Image.FLIP_LEFT_RIGHT)
    if scale != 1.0:
        bat = bat.resize((int(bat.width * scale), int(bat.height * scale)), Image.LANCZOS)
    # 硬 alpha
    ba = np.array(bat)
    alpha = ba[..., 3]
    fg = alpha > 60
    ba[..., 3] = np.where(fg, 255, 0)
    return Image.fromarray(ba)


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

    # 顶弧字
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


def compose_one(tag, big, sub, arc, mode, scale, clean_base):
    bat, bat_center = extract_bat_patch()
    bat = transform_bat(bat, mode, scale)

    canvas = clean_base.convert("RGBA")
    cx, cy = bat_center
    paste_x = cx - bat.width // 2
    paste_y = cy - bat.height // 2
    canvas.paste(bat, (paste_x, paste_y), bat)
    rgb = canvas.convert("RGB")

    small = rgb.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    return burn_text(small, arc, big, sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base_path = COMFY_INPUT / "v276_clean_base.png"
    if not base_path.exists() or args.force:
        clean = make_clean_base()
        clean.save(base_path, quality=98)
        print(f"[v276] clean base -> {base_path}")
    else:
        clean = Image.open(base_path).convert("RGB")

    files = []
    for tag, big, sub, arc, mode, scale in SUBJECT_VARIANTS:
        final = compose_one(tag, big, sub, arc, mode, scale, clean)
        out = JOB / f"v276_{tag}_final.png"
        final.save(out, quality=95)
        files.append(out)
        print(f"  {tag} -> {out.name}")

    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB"), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB"), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB"), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v276.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")


if __name__ == "__main__":
    main()
