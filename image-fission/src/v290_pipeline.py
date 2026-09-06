"""
v290 — 修复 v289 弧带/底字区域色块横痕。

核心修复：
  1. 弧带 + 主/副字区域全部用 cv2.inpaint (NS) 去字，原图局部背景色自然重建，不再有硬色块
  2. bat mask 保护：所有去字区域减去 bat boundingRect + 圆形保护区，避免误伤蝙蝠翅膀
  3. inpaint 后再提取 bat mask（确保蝙蝠边界按原始轮廓走）
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
JOB = PROJECT / "jobs" / "v290"
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
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",      290001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",       290101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK",    290201),
]


def _ring_sector_mask(h, w, cx, cy, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) - a0) % 360
    sweep = (a1 - a0) % 360
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    in_angle = ang <= sweep
    return in_ring & in_angle


def _detect_text_pixels(arr, region_mask, threshold=12):
    h, w = arr.shape[:2]
    if region_mask.sum() < 50:
        return np.zeros((h, w), dtype=np.uint8)
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


def _detect_bat_mask(arr):
    """基于原图检测蝙蝠 mask（深紫剪影）。"""
    h, w = arr.shape[:2]
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(arr.astype(np.float32), axis=2)
    dark = (maxc < 80) & (stdc < 25)
    mask = dark.astype(np.uint8) * 255
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge + 30, 255, -1)
    mask = cv2.bitwise_and(mask, badge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(sizes))
        mask = (labels == keep).astype(np.uint8) * 255
    return mask


def _bat_protect_mask(bat_mask):
    """蝙蝠保护区域：bat boundingRect + 扩展 + 圆形 badge。"""
    h, w = bat_mask.shape
    ys, xs = np.where(bat_mask > 127)
    if len(xs) == 0:
        return np.zeros((h, w), dtype=bool)
    protect = np.zeros((h, w), dtype=bool)
    bx0, bx1 = max(0, xs.min() - 40), min(w, xs.max() + 40)
    by0, by1 = max(0, ys.min() - 30), min(h, ys.max() + 60)
    protect[by0:by1, bx0:bx1] = True
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge + 60, 255, -1)
    protect = protect | (badge > 0)
    return protect


def _inpaint_text(arr, text_mask, radius=8.0):
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    inpaint_mask = (text_mask > 127).astype(np.uint8) * 255
    return cv2.inpaint(arr_u8, inpaint_mask, radius, cv2.INPAINT_NS)


def make_text_free_source():
    """按以下顺序抹字：1) 弧带 → 2) 主/副字 → 3) Est/1862/三角；全程 bat 保护。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]

    bat_mask = _detect_bat_mask(arr)
    protect = _bat_protect_mask(bat_mask)

    # 弧带 inpaint
    arc_region = _ring_sector_mask(h, w, cx, cy, ring_outer - 95, ring_outer + 65, 185, 355)
    inner_exclude = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 30, 0, 360)
    arc_region = arc_region & (~inner_exclude)
    arc_region_safe = arc_region & (~protect)
    arc_text = _detect_text_pixels(arr, arc_region_safe, threshold=8)
    arc_text = cv2.dilate(arc_text, np.ones((7, 7), np.uint8), iterations=3)
    arc_text = cv2.morphologyEx(arc_text, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if arc_text.sum() > 100:
        arr_u8 = _inpaint_text(arr, arc_text, radius=8.0)
        arr = arr_u8.astype(np.float32)
        print(f"[arc] inpaint mask={int(arc_text.sum()/255)}px")

    # 主/副/Est/1862/三角矩形 inpaint
    rects = [
        (1000, 1185, 80, 1470),
        (1185, 1380, 240, 1310),
        (700,  910,  240,  680),
        (700,  910,  870, 1310),
        (1450, 1550, 620,  930),
    ]
    for y0, y1, x0, x1 in rects:
        region = np.zeros((h, w), dtype=np.uint8)
        region[y0:y1, x0:x1] = 255
        region_safe = region & (~protect)
        if region_safe.sum() < 500:
            continue
        text_pixels = _detect_text_pixels(arr, region_safe > 0, threshold=10)
        text_pixels = cv2.dilate(text_pixels, np.ones((9, 9), np.uint8), iterations=2)
        text_pixels = cv2.morphologyEx(text_pixels, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        text_pixels[bat_mask > 127] = 0
        if text_pixels.sum() < 50:
            continue
        arr_u8 = _inpaint_text(arr, text_pixels, radius=7.0)
        arr = arr_u8.astype(np.float32)
        print(f"[rect] {y0}-{y1},{x0}-{x1} mask={int(text_pixels.sum()/255)}px")

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def extract_bat(text_free_img):
    src = np.array(text_free_img).astype(np.float32)
    h, w = src.shape[:2]
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge, 255, -1)
    R, G, B = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(src, axis=2)
    dark = (maxc < 80) & (stdc < 25)
    mask = dark.astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, badge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(sizes))
        mask = (labels == keep).astype(np.uint8) * 255
    bat_pixels = np.zeros_like(src)
    bat_pixels[mask > 0] = (0, 0, 0)
    return bat_pixels, mask


def _warp_bat_by_wings(bat_pixels, mask, left_delta, right_delta, shoulder_l, shoulder_r, falloff=110):
    h, w = mask.shape
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    dl = np.sqrt((xs - shoulder_l[0]) ** 2 + (ys - shoulder_l[1]) ** 2)
    dr = np.sqrt((xs - shoulder_r[0]) ** 2 + (ys - shoulder_r[1]) ** 2)
    wl = np.clip((dl - 25) / falloff, 0, 1) * (xs < cx).astype(np.float32)
    wr = np.clip((dr - 25) / falloff, 0, 1) * (xs > cx).astype(np.float32)
    dx_map = wl * left_delta[0] + wr * right_delta[0]
    dy_map = wl * left_delta[1] + wr * right_delta[1]
    x_map = xs - dx_map
    y_map = ys - dy_map
    warped_mask = cv2.remap(mask, x_map.astype(np.float32), y_map.astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_pixels = cv2.remap(bat_pixels, x_map.astype(np.float32), y_map.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge, 255, -1)
    warped_mask = cv2.bitwise_and(warped_mask, badge)
    warped_pixels = warped_pixels * (badge[:, :, None] > 0)
    _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
    warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)
    alpha = warped_mask.astype(np.float32) / 255.0
    return warped_pixels, alpha


def _find_wing_landmarks(mask):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    upper = ys < cy
    if upper.sum() < 20:
        upper = ys < cy + 80
    left_idx = xs[upper].argmin()
    right_idx = xs[upper].argmax()
    left_tip = (int(xs[upper][left_idx]), int(ys[upper][left_idx]))
    right_tip = (int(xs[upper][right_idx]), int(ys[upper][right_idx]))
    shoulder_y = int((left_tip[1] + cy) * 0.55)
    row = mask[shoulder_y, :]
    row_idx = np.where(row > 127)[0]
    if len(row_idx) >= 2:
        left_shoulder = (int(row_idx[row_idx < cx].max()) if np.any(row_idx < cx) else cx - 80, shoulder_y)
        right_shoulder = (int(row_idx[row_idx > cx].min()) if np.any(row_idx > cx) else cx + 80, shoulder_y)
    else:
        left_shoulder = (cx - 90, shoulder_y)
        right_shoulder = (cx + 90, shoulder_y)
    return {"left_tip": left_tip, "right_tip": right_tip,
            "left_shoulder": left_shoulder, "right_shoulder": right_shoulder}


def make_variants(text_free_img):
    bat_pixels, mask = extract_bat(text_free_img)
    lm = _find_wing_landmarks(mask)
    if lm is None:
        raise RuntimeError("未找到蝙蝠 landmark")

    variants = {}
    variants["up"] = _warp_bat_by_wings(bat_pixels, mask,
        left_delta=(10, -55), right_delta=(-10, -55),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)
    variants["spread"] = _warp_bat_by_wings(bat_pixels, mask,
        left_delta=(-55, -15), right_delta=(55, -15),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)
    variants["fold"] = _warp_bat_by_wings(bat_pixels, mask,
        left_delta=(-15, 55), right_delta=(15, 55),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)
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

    src_path = COMFY_INPUT / "v290_text_free_source.png"

    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v290] text-free source -> {src_path}")
    else:
        print(f"[v290] reuse text-free source")

    text_free = Image.open(src_path).convert("RGB")
    bg_arr = np.array(text_free).astype(np.float32)
    variants = make_variants(text_free)

    files = []
    for tag, big, sub, arc, _ in SUBJECT_VARIANTS:
        warped_pixels, alpha = variants[tag]
        composed = bg_arr * (1 - alpha[:, :, None]) + warped_pixels * alpha[:, :, None]
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v290_{tag}_composed.png", quality=95)

        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v290_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"[v290] {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v290.png"
    grid.save(gp, quality=92)
    print(f"[v290] grid -> {gp}")


if __name__ == "__main__":
    main()
