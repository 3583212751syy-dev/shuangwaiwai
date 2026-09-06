"""
v286 — 放弃让 SDXL 重画蝙蝠，改为程序直接对原蝙蝠做姿态形变，100% 避免缺失/像素/颜色问题。

修复方向：
  1. 主体缺失/像素缺失 → 不生成新蝙蝠，直接仿射/透视形变原蝙蝠剪影，保证完整实心黑
  2. 背景颜色改变 → 背景像素完全不动
  3. 文字背景遮挡 → 弧带用径向渐变填充匹配横幅，主/副字用局部背景色填充

核心：
  - 提取原蝙蝠 mask，三种仿射形变（up/spread/fold）
  - 用原蝙蝠像素按形变后 mask 贴回，背景 100% 保留
  - 不再调用 ComfyUI 生成蝙蝠，只烧新字
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "v286"
JOB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "src"))
import arc_text

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

TARGET_W, TARGET_H = 1024, 1280
INK = (26, 10, 31)

cx, cy, r_badge = 776, 746, 300
ring_outer = 421

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  286001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   286101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 286201),
]


def _ring_sector_mask(h, w, cx, cy, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) - a0) % 360
    sweep = (a1 - a0) % 360
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    in_angle = ang <= sweep
    return in_ring & in_angle


def _detect_text_pixels(arr, region_mask, threshold=35):
    h, w = arr.shape[:2]
    region = region_mask.astype(np.uint8) * 255
    dilated = cv2.dilate(region, np.ones((12, 12), np.uint8), iterations=1)
    eroded = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=1)
    ring = (dilated > 0) & (eroded == 0)
    if ring.sum() < 50:
        bg = np.median(arr.reshape(-1, 3), axis=0)
    else:
        bg = np.median(arr[ring].reshape(-1, 3), axis=0)
    maxc = np.max(arr, axis=2)
    bg_max = np.max(bg)
    dark = (maxc < bg_max - threshold) & (region_mask > 0)
    dark = dark.astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return dark


def _local_median_fill(arr, text_mask, k=15):
    h, w = arr.shape[:2]
    text_u8 = (text_mask > 127).astype(np.uint8) * 255
    inv = 255 - text_u8
    bg_sum = np.zeros_like(arr, dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    for dy in range(-k, k + 1, 3):
        for dx in range(-k, k + 1, 3):
            if dy == 0 and dx == 0:
                continue
            sy, ey = max(0, dy), min(h, h + dy)
            sx, ex = max(0, dx), min(w, w + dx)
            dy_src, ey_src = sy - dy, ey - dy
            dx_src, ex_src = sx - dx, ex - dx
            shifted = np.zeros_like(arr)
            shifted[sy:ey, sx:ex] = arr[dy_src:ey_src, dx_src:ex_src]
            shifted_inv = np.zeros((h, w), dtype=np.uint8)
            shifted_inv[sy:ey, sx:ex] = inv[dy_src:ey_src, dx_src:ex_src]
            valid = (shifted_inv > 0).astype(np.float32)[:, :, None]
            bg_sum += shifted * valid
            weight += valid
    weight = np.maximum(weight, 1.0)
    bg = bg_sum / weight
    empty = (weight[:, :, 0] < 0.5)
    if empty.any():
        global_bg = np.median(arr.reshape(-1, 3), axis=0)
        bg[empty] = global_bg
    return bg


def _fill_text_region_local(arr, region_mask, feather=15, threshold=18):
    h, w = arr.shape[:2]
    text_mask = _detect_text_pixels(arr, region_mask, threshold=threshold)
    text_mask = cv2.dilate(text_mask, np.ones((5, 5), np.uint8), iterations=2)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    bg = _local_median_fill(arr, text_mask, k=21)
    soft = cv2.GaussianBlur(text_mask.astype(np.float32), (feather*2+1, feather*2+1), 0) / 255.0
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - soft) + bg[:, :, c] * soft
    return arr


def _fill_text_pixels_uniform(arr, text_mask, feather=15, noise_scale=0.35):
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
        std = np.clip(pixels.std(axis=0), 1.5, 5.0)

    filled = arr.copy()
    filled[text_mask_u8 > 0] = bg

    noise = np.random.normal(0, std * noise_scale, size=(h, w, 3)).astype(np.float32)
    filled = filled + noise * (text_mask_u8[:, :, None] > 0)
    filled = cv2.GaussianBlur(filled.astype(np.uint8), (5, 5), 0).astype(np.float32)
    filled = np.clip(filled, 0, 255)

    m_blur = cv2.GaussianBlur(text_mask_u8.astype(np.float32), (feather*2+1, feather*2+1), 0) / 255.0
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - m_blur) + filled[:, :, c] * m_blur
    return arr


def make_text_free_source():
    """去字：弧带局部中值；主/副/Est/1862 均匀填充。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]

    arc_region = _ring_sector_mask(h, w, cx, cy, ring_outer - 95, ring_outer + 65, 185, 355)
    inner_exclude = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 30, 0, 360)
    arc_region = arc_region & (~inner_exclude)
    arr = _fill_text_region_local(arr, arc_region, feather=21, threshold=8)

    rects = [
        (1000, 1185, 80, 1470, 25),
        (1185, 1380, 240, 1310, 25),
        (700,  910,  240,  680, 25),
        (700,  910,  870, 1310, 25),
        (1450, 1550, 620,  930, 25),
    ]
    for y0, y1, x0, x1, feather in rects:
        region = np.zeros((h, w), dtype=np.uint8)
        region[y0:y1, x0:x1] = 255
        region = cv2.dilate(region, np.ones((7, 7), np.uint8), iterations=1)
        arr = _fill_text_pixels_uniform(arr, region, feather=feather)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def extract_bat():
    """从 text_free_source 提取原蝙蝠像素和 mask。"""
    src = np.array(make_text_free_source()).astype(np.float32)
    h, w = src.shape[:2]
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
    mask = cv2.dilate(mask, np.ones((6, 6), np.uint8), iterations=1)

    # 蝙蝠像素：从原图取（text_free_source 里蝙蝠还在）
    bat_pixels = src.copy()
    bat_pixels[mask == 0] = 0
    return bat_pixels, mask


def _warp_bat(bat_pixels, mask, sx, sy, rot_deg=0.0, fold_factor=0.0):
    """以 (cx,cy) 为中心对蝙蝠做仿射/透视形变。"""
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    M[0, 0] *= sx
    M[0, 1] *= sx
    M[1, 0] *= sy
    M[1, 1] *= sy
    M[0, 2] = cx - sx * (cx * math.cos(math.radians(rot_deg)) - cy * math.sin(math.radians(rot_deg)))
    M[1, 2] = cy - sy * (cx * math.sin(math.radians(rot_deg)) + cy * math.cos(math.radians(rot_deg)))

    warped_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_pixels = cv2.warpAffine(bat_pixels, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    if fold_factor != 0.0:
        ys = np.arange(h).reshape(-1, 1).astype(np.float32)
        factor = 1.0 + fold_factor * np.clip((ys - cy) / (h - cy), 0, 1)
        x_map = (np.arange(w).astype(np.float32)[None, :] - cx) / factor + cx
        y_map = np.tile(ys, (1, w))
        warped_mask = cv2.remap(warped_mask, x_map.astype(np.float32), y_map.astype(np.float32),
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_pixels = cv2.remap(warped_pixels, x_map.astype(np.float32), y_map.astype(np.float32),
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 限制在 inner badge 内
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge, 255, -1)
    warped_mask = cv2.bitwise_and(warped_mask, badge)
    warped_pixels = warped_pixels * (badge[:, :, None] > 0)

    # 稍微羽化边缘
    warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 0)
    alpha = warped_mask.astype(np.float32) / 255.0
    return warped_pixels, alpha


def make_variants():
    """生成三种姿态蝙蝠。"""
    bat_pixels, mask = extract_bat()
    variants = {}
    variants["up"] = _warp_bat(bat_pixels, mask, sx=0.78, sy=1.28, rot_deg=-2)
    variants["spread"] = _warp_bat(bat_pixels, mask, sx=1.45, sy=0.68, rot_deg=0)
    variants["fold"] = _warp_bat(bat_pixels, mask, sx=1.12, sy=0.88, rot_deg=1, fold_factor=0.25)
    return variants


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

    src_path = COMFY_INPUT / "v286_text_free_source.png"

    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v286] text-free source -> {src_path}")
    else:
        print(f"[v286] reuse text-free source")

    text_free = Image.open(src_path).convert("RGB")
    bg_arr = np.array(text_free).astype(np.float32)
    variants = make_variants()

    files = []
    for tag, big, sub, arc, _ in SUBJECT_VARIANTS:
        warped_pixels, alpha = variants[tag]
        composed = bg_arr * (1 - alpha[:, :, None]) + warped_pixels * alpha[:, :, None]
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v286_{tag}_composed.png", quality=95)

        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v286_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"[v286] {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v286.png"
    grid.save(gp, quality=92)
    print(f"[v286] grid -> {gp}")


if __name__ == "__main__":
    main()
