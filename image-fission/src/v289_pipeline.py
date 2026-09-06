"""
v289 — 修复 v288 蝙蝠形变 unnatural 问题。

核心：放弃整蝙蝠的仿射/透视拉伸，改为"肩部固定 + 翅尖位移"的非均匀形变：
  - 身体/躯干 100% 不动
  - 只让左右翅尖按自然方向移动（上扬 / 横展 / 下垂）
  - 位移从肩部向翅尖平滑过渡，避免硬边与比例崩坏
  - 弧带/文字部分沿用 v288 的径向渐变 + 局部背景色填充
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
JOB = PROJECT / "jobs" / "v289"
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
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  289001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   289101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 289201),
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


def _fill_arc_gradient(arr, arc_region, text_mask):
    """用径向渐变填充弧带：按半径取非文字像素中值，然后高斯平滑。"""
    h, w = arr.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.int32)

    ys_idx, xs_idx = np.where(arc_region)
    r_vals = r[ys_idx, xs_idx]
    r_min, r_max = r_vals.min(), r_vals.max()

    non_text = arc_region & (text_mask == 0)
    color_by_r = {}
    for rr in range(r_min, r_max + 1):
        mask_r = non_text & (r == rr)
        if mask_r.sum() > 0:
            color_by_r[rr] = np.median(arr[mask_r].reshape(-1, 3), axis=0)
        elif rr - 1 in color_by_r:
            color_by_r[rr] = color_by_r[rr - 1]
        elif rr + 1 in color_by_r:
            color_by_r[rr] = color_by_r[rr + 1]

    rrs = sorted(color_by_r.keys())
    for rr in range(r_min, r_max + 1):
        if rr not in color_by_r:
            lower = [x for x in rrs if x < rr]
            upper = [x for x in rrs if x > rr]
            if lower and upper:
                l, u = lower[-1], upper[0]
                t = (rr - l) / (u - l)
                color_by_r[rr] = color_by_r[l] * (1 - t) + color_by_r[u] * t
            elif lower:
                color_by_r[rr] = color_by_r[lower[-1]]
            elif upper:
                color_by_r[rr] = color_by_r[upper[0]]

    gradient = np.zeros_like(arr)
    for rr in range(r_min, r_max + 1):
        gradient[r == rr] = color_by_r[rr]

    arc_u8 = arc_region.astype(np.uint8) * 255
    arc_soft = cv2.GaussianBlur(arc_u8.astype(np.float32), (25, 25), 0) / 255.0

    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - arc_soft) + gradient[:, :, c] * arc_soft
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
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]

    arc_region = _ring_sector_mask(h, w, cx, cy, ring_outer - 95, ring_outer + 65, 185, 355)
    inner_exclude = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 30, 0, 360)
    arc_region = arc_region & (~inner_exclude)
    arc_text_mask = _detect_text_pixels(arr, arc_region, threshold=8)
    arc_text_mask = cv2.dilate(arc_text_mask, np.ones((7, 7), np.uint8), iterations=3)
    arc_text_mask = cv2.morphologyEx(arc_text_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    arr = _fill_arc_gradient(arr, arc_region, arc_text_mask)

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
    mask = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)

    bat_pixels = np.zeros_like(src)
    bat_pixels[mask > 0] = (0, 0, 0)
    return bat_pixels, mask


def _find_wing_landmarks(mask):
    """从蝙蝠 mask 找出左右翅尖与左右肩点（身体保持不动的锚点）。"""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None

    # 上边缘点（翅尖候选）
    upper = ys < cy
    if upper.sum() < 20:
        upper = ys < cy + 80

    left_idx = xs[upper].argmin()
    right_idx = xs[upper].argmax()
    left_tip = (int(xs[upper][left_idx]), int(ys[upper][left_idx]))
    right_tip = (int(xs[upper][right_idx]), int(ys[upper][right_idx]))

    # 肩点：身体宽度处、y 在翅尖与质心之间
    shoulder_y = int((left_tip[1] + cy) * 0.55)
    row = mask[shoulder_y, :]
    row_idx = np.where(row > 127)[0]
    if len(row_idx) >= 2:
        left_shoulder = (int(row_idx[row_idx < cx].max()) if np.any(row_idx < cx) else cx - 80, shoulder_y)
        right_shoulder = (int(row_idx[row_idx > cx].min()) if np.any(row_idx > cx) else cx + 80, shoulder_y)
    else:
        left_shoulder = (cx - 90, shoulder_y)
        right_shoulder = (cx + 90, shoulder_y)

    return {
        "left_tip": left_tip, "right_tip": right_tip,
        "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
    }


def _warp_bat_by_wings(bat_pixels, mask, left_delta, right_delta, shoulder_l, shoulder_r, falloff=110):
    """
    肩部固定，只移动翅尖。
    left_delta/right_delta: 翅尖目标位移 (dx, dy)
    falloff: 位移从肩部到翅尖的过渡距离
    """
    h, w = mask.shape
    ys, xs = np.mgrid[:h, :w]
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    # 到左右肩的距离
    dl = np.sqrt((xs - shoulder_l[0]) ** 2 + (ys - shoulder_l[1]) ** 2)
    dr = np.sqrt((xs - shoulder_r[0]) ** 2 + (ys - shoulder_r[1]) ** 2)

    # 权重：肩部附近为 0，向翅尖线性增至 1；只用同侧翅膀
    wl = np.clip((dl - 25) / falloff, 0, 1) * (xs < cx).astype(np.float32)
    wr = np.clip((dr - 25) / falloff, 0, 1) * (xs > cx).astype(np.float32)

    # 位移场
    dx_map = wl * left_delta[0] + wr * right_delta[0]
    dy_map = wl * left_delta[1] + wr * right_delta[1]

    x_map = xs - dx_map
    y_map = ys - dy_map

    warped_mask = cv2.remap(mask, x_map.astype(np.float32), y_map.astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_pixels = cv2.remap(bat_pixels, x_map.astype(np.float32), y_map.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 限制在 badge 圆内
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge, 255, -1)
    warped_mask = cv2.bitwise_and(warped_mask, badge)
    warped_pixels = warped_pixels * (badge[:, :, None] > 0)

    _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
    warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)
    alpha = warped_mask.astype(np.float32) / 255.0
    return warped_pixels, alpha


def make_variants():
    bat_pixels, mask = extract_bat()
    lm = _find_wing_landmarks(mask)
    if lm is None:
        raise RuntimeError("无法提取蝙蝠 landmark")

    variants = {}
    # up: 翅尖上扬；spread: 翅尖外展；fold: 翅尖下垂
    variants["up"] = _warp_bat_by_wings(
        bat_pixels, mask,
        left_delta=(10, -55), right_delta=(-10, -55),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)

    variants["spread"] = _warp_bat_by_wings(
        bat_pixels, mask,
        left_delta=(-55, -15), right_delta=(55, -15),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)

    variants["fold"] = _warp_bat_by_wings(
        bat_pixels, mask,
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

    src_path = COMFY_INPUT / "v289_text_free_source.png"

    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v289] text-free source -> {src_path}")
    else:
        print(f"[v289] reuse text-free source")

    text_free = Image.open(src_path).convert("RGB")
    bg_arr = np.array(text_free).astype(np.float32)
    variants = make_variants()

    files = []
    for tag, big, sub, arc, _ in SUBJECT_VARIANTS:
        warped_pixels, alpha = variants[tag]
        composed = bg_arr * (1 - alpha[:, :, None]) + warped_pixels * alpha[:, :, None]
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v289_{tag}_composed.png", quality=95)

        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v289_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"[v289] {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v289.png"
    grid.save(gp, quality=92)
    print(f"[v289] grid -> {gp}")


if __name__ == "__main__":
    main()
